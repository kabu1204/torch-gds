#include <cstdio>
#include <vector>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

int get_thread_core() {
    int id = -1;
	asm volatile (
		"rdtscp\n\t"
		"mov %%ecx, %0\n\t":
		"=r" (id) :: "%rax", "%rcx", "%rdx");
	return id & 0xFFF;
}

struct {
	int (*pcm_c_build_core_event)(uint8_t id, const char * argv);
	int (*pcm_c_init)();
	void (*pcm_c_start)();
	void (*pcm_c_stop)();
	uint64_t (*pcm_c_get_cycles)(uint32_t core_id);
	uint64_t (*pcm_c_get_instr)(uint32_t core_id);
	uint64_t (*pcm_c_get_core_event)(uint32_t core_id, uint32_t event_id);
} PCM; // lgtm [cpp/short-global-name]
