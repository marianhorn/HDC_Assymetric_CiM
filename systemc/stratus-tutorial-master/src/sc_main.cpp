#include "ringbuffer_tb.h"

ringbuffer_tb *tb;

extern void esc_elaborate()
{
	tb = new ringbuffer_tb("ringbuffer_tb");
}

extern void esc_cleanup()
{
	delete tb;
}

int sc_main(int argc, char *argv[])
{
	esc_initialize(argc, argv);
	esc_elaborate();

	sc_clock 		clk("clk", 10, SC_NS);
	sc_signal<bool> 	rst_n;

	tb->clk(clk);
	tb->rst_n(rst_n);

	rst_n.write(false);

	sc_start(30, SC_NS);

	rst_n.write(true);

	sc_start(1, SC_US);

	return 0;
}
