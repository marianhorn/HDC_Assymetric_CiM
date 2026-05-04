SYSTEMC_HOME ?= /usr/local/systemc
CXX ?= g++
TARGET ?= systemc_hdc_tb
SRC_DIR := src
BUILD_DIR := build

SYSTEMC_INCLUDE := $(SYSTEMC_HOME)/include
SYSTEMC_LIB_DIR ?= $(firstword $(wildcard $(SYSTEMC_HOME)/lib-linux64) $(wildcard $(SYSTEMC_HOME)/lib64) $(wildcard $(SYSTEMC_HOME)/lib))

ifeq ($(strip $(SYSTEMC_LIB_DIR)),)
$(error Could not find a SystemC library directory under $(SYSTEMC_HOME). Set SYSTEMC_LIB_DIR explicitly.)
endif

CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic -I$(SRC_DIR) -I. -I$(SYSTEMC_INCLUDE)
LDFLAGS ?= -L$(SYSTEMC_LIB_DIR) -Wl,-rpath,$(SYSTEMC_LIB_DIR) -lsystemc -lm

SOURCES := $(SRC_DIR)/hdc_memory.cpp $(SRC_DIR)/hdc_accelerator.cpp $(SRC_DIR)/controller.cpp $(SRC_DIR)/foot_dataset_loader.cpp $(SRC_DIR)/tb_systemc.cpp
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
HEADERS := $(SRC_DIR)/systemc_types.h $(SRC_DIR)/hdc_memory.h $(SRC_DIR)/hdc_accelerator.h $(SRC_DIR)/controller.h $(SRC_DIR)/foot_dataset_loader.h $(SRC_DIR)/config_systemc.h

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
