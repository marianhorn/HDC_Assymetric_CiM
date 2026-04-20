SYSTEMC_HOME ?= /usr/local/systemc
CXX ?= g++
TARGET ?= systemc_hdc_tb

SYSTEMC_INCLUDE := $(SYSTEMC_HOME)/include
SYSTEMC_LIB_DIR ?= $(firstword $(wildcard $(SYSTEMC_HOME)/lib-linux64) $(wildcard $(SYSTEMC_HOME)/lib64) $(wildcard $(SYSTEMC_HOME)/lib))

ifeq ($(strip $(SYSTEMC_LIB_DIR)),)
$(error Could not find a SystemC library directory under $(SYSTEMC_HOME). Set SYSTEMC_LIB_DIR explicitly.)
endif

CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic -I. -I$(SYSTEMC_INCLUDE)
LDFLAGS ?= -L$(SYSTEMC_LIB_DIR) -Wl,-rpath,$(SYSTEMC_LIB_DIR) -lsystemc -lm

SOURCES := hdc_systemc.cpp tb_systemc.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp hdc_systemc.h config_systemc.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
