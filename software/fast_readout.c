/*
 * fast_readout.c — C-accelerated FPGA feature readout
 *
 * Eliminates Python MMIO overhead (~30μs/read → ~0.3μs/read).
 * Reads Layer-2 features from BRAM via AXI-Lite registers.
 *
 * Compile on PYNQ:
 *   gcc -shared -fPIC -O2 -o fast_readout.so fast_readout.c
 */

#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

/* AXI-Lite register offsets */
#define REG_CONTROL     0x00
#define REG_STATUS      0x04
#define REG_OUTPUT_CH   0x20
#define REG_OUTPUT_ADDR 0x24
#define REG_OUTPUT_DATA 0x28

/*
 * read_features_full:
 *   Reads all 256 values for each of n_ch channels.
 *   Output: flat uint8 array of size n_ch * 256.
 *
 *   base     — mmap'd pointer to AXI-Lite register base
 *   out      — output buffer (n_ch * 256 bytes)
 *   n_ch     — number of channels to read (typically 64)
 *   ch_off   — BRAM channel offset (typically 48)
 */
void read_features_full(volatile uint32_t *base, uint8_t *out,
                        int n_ch, int ch_off)
{
    int ch, a, idx = 0;
    for (ch = 0; ch < n_ch; ch++) {
        base[REG_OUTPUT_CH / 4] = (uint32_t)(ch_off + ch);
        for (a = 0; a < 256; a++) {
            base[REG_OUTPUT_ADDR / 4] = (uint32_t)a;
            (void)base[REG_OUTPUT_DATA / 4];           /* trigger BRAM */
            out[idx++] = (uint8_t)(base[REG_OUTPUT_DATA / 4] & 0xFF);
        }
    }
}

/*
 * read_features_sub:
 *   Reads only specified addresses (bin centers) per channel.
 *   Much faster — 16 values/channel instead of 256.
 *
 *   addrs    — array of BRAM addresses to read (length n_addrs)
 *   n_addrs  — number of addresses per channel (typically 16)
 */
void read_features_sub(volatile uint32_t *base, uint8_t *out,
                       int n_ch, int ch_off,
                       const int *addrs, int n_addrs)
{
    int ch, i, idx = 0;
    for (ch = 0; ch < n_ch; ch++) {
        base[REG_OUTPUT_CH / 4] = (uint32_t)(ch_off + ch);
        for (i = 0; i < n_addrs; i++) {
            base[REG_OUTPUT_ADDR / 4] = (uint32_t)addrs[i];
            (void)base[REG_OUTPUT_DATA / 4];
            out[idx++] = (uint8_t)(base[REG_OUTPUT_DATA / 4] & 0xFF);
        }
    }
}

/*
 * send_image_start:
 *   Writes control registers to start inference, then polls until done.
 *   Returns 0 on success, -1 on timeout.
 *
 *   timeout_us — timeout in microseconds
 */
int start_and_wait(volatile uint32_t *base, int timeout_us)
{
    int elapsed = 0;
    base[REG_CONTROL / 4] = 0x01;  /* start */

    while (elapsed < timeout_us) {
        uint32_t status = base[REG_STATUS / 4];
        if ((status >> 1) & 1)
            return 0;  /* done */
        /* Busy wait ~1μs */
        volatile int x = 0;
        for (int i = 0; i < 50; i++) x += i;
        elapsed += 1;
    }
    return -1;  /* timeout */
}

/*
 * open_devmem:
 *   Opens /dev/mem and returns mmap'd pointer to given physical address.
 *   Returns NULL on failure.
 */
volatile uint32_t *open_devmem(uint32_t phys_addr, uint32_t size)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0;

    uint32_t page = phys_addr & ~0xFFF;
    uint32_t offset = phys_addr - page;

    void *map = mmap(0, size + offset, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, page);
    close(fd);

    if (map == MAP_FAILED) return 0;
    return (volatile uint32_t *)((uint8_t *)map + offset);
}
