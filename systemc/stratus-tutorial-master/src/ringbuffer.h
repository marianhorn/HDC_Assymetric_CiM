#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <systemc.h>
#include <stratus_hls.h>

#define RINGBUFFER_DATA_WIDTH 32
#define RINGBUFFER_SIZE 16
#define RINGBUFFER_POSITION_WIDTH 4

SC_MODULE(ringbuffer)
{
	sc_in<bool>				clk;
	sc_in<bool>				rst_n;

	sc_in<sc_logic> 			write_en;
	sc_in<sc_bv<RINGBUFFER_DATA_WIDTH>> 	write_data;

	sc_in<sc_logic>				read_en;
	sc_out<sc_bv<RINGBUFFER_DATA_WIDTH>>	read_data;

	sc_out<sc_logic>			empty;
	sc_out<sc_logic>			full;

	SC_CTOR(ringbuffer)
	{
		SC_CTHREAD(read_write, clk.pos())
		reset_signal_is(rst_n, false);
		
		HLS_MAP_TO_MEMORY(ringbuffer_data);
	}

	void read_write();

	sc_uint<RINGBUFFER_POSITION_WIDTH> read_pos;
	sc_uint<RINGBUFFER_POSITION_WIDTH> write_pos;

	sc_uint<RINGBUFFER_POSITION_WIDTH + 1> num_elements;

	sc_bv<RINGBUFFER_DATA_WIDTH> ringbuffer_data[RINGBUFFER_SIZE];
};

#endif
