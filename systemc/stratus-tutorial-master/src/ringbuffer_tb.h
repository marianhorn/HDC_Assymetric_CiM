#ifndef RINGBUFFER_TB_H
#define RINGBUFFER_TB_H

#include <systemc.h>

#include "ringbuffer_wrap.h"

#define RINGBUFFER_DATA_WIDTH 32

SC_MODULE(ringbuffer_tb)
{
	sc_in<bool> clk;
	sc_in<bool> rst_n;

	SC_CTOR(ringbuffer_tb)
	{

		SC_CTHREAD(read_write, clk.pos());
		reset_signal_is(rst_n, false);

		uut = new ringbuffer_wrapper("ringbuffer_0");
		
		uut->clk(clk);
		uut->rst_n(rst_n);

		uut->write_en(write_en);
		uut->write_data(write_data);

		uut->read_en(read_en);
		uut->read_data(read_data);

		uut->empty(empty);
		uut->full(full);

		
	}
	
	~ringbuffer_tb()
	{
		delete uut;
	}

	void read_write();

	sc_signal<sc_logic>			write_en;
	sc_signal<sc_bv<RINGBUFFER_DATA_WIDTH>>	write_data;

	sc_signal<sc_logic>			read_en;
	sc_signal<sc_bv<RINGBUFFER_DATA_WIDTH>>	read_data;

	sc_signal<sc_logic>			empty;
	sc_signal<sc_logic>			full;

	sc_uint<RINGBUFFER_DATA_WIDTH>		write_val;
	sc_uint<RINGBUFFER_DATA_WIDTH>		read_val;

	ringbuffer_wrapper *uut;
};

#endif
