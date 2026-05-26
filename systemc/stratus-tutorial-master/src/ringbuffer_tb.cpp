#include "ringbuffer_tb.h"

void ringbuffer_tb::read_write()
{
	read_en.write(sc_logic('0'));

	write_en.write(sc_logic('0'));
	write_data.write((sc_bv<RINGBUFFER_DATA_WIDTH>) 0);
	write_val = 1;

	wait();

	while(1)
	{
		if(full.read() == sc_logic('0'))
		{
			cout << sc_time_stamp(); 
			cout << "\twrote " << write_val << endl;

			write_en.write(sc_logic('1'));
			write_data.write((sc_bv<RINGBUFFER_DATA_WIDTH>) write_val);
			write_val++;

		}
		else
		{
			cout << sc_time_stamp(); 
			cout << "\tringbuffer full" << endl;

			write_en.write(sc_logic('0'));
		}

		wait();
	}
}
