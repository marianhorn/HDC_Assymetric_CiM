#include "ringbuffer.h"

void ringbuffer::read_write()
{
	read_pos = 0;
	write_pos = 0;

	num_elements = 0;

	empty.write(sc_logic('1'));
	full.write(sc_logic('0'));

	wait();

	while(1)
	{
		if(read_en.read() == sc_logic('1'))
		{
			read_data.write(ringbuffer_data[read_pos]);
			read_pos++;

			num_elements--;
			
		}
		
		if(write_en.read() == sc_logic('1'))
		{
			ringbuffer_data[write_pos] = write_data.read();
			write_pos++;

			num_elements++;
		}

		if(num_elements == 0) empty.write(sc_logic('1'));
		else empty.write(sc_logic('0'));

		if(num_elements == RINGBUFFER_SIZE) full.write(sc_logic('1'));
		else full.write(sc_logic('0'));
	
		wait();
	}
}
