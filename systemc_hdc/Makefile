SYSTEMC_HOME ?= $(firstword $(wildcard /home/zora/systemc-install) $(wildcard /usr/local/systemc))
CXX ?= g++
TARGET ?= systemc_hdc_tb
GOLDEN_TARGET ?= systemc_hdc_golden_regression
SRC_DIR := src
BUILD_DIR := build

SYSTEMC_INCLUDE := $(SYSTEMC_HOME)/include
SYSTEMC_LIB_DIR ?= $(firstword $(wildcard $(SYSTEMC_HOME)/lib-linux64) $(wildcard $(SYSTEMC_HOME)/lib64) $(wildcard $(SYSTEMC_HOME)/lib))

ifeq ($(strip $(SYSTEMC_LIB_DIR)),)
$(error Could not find a SystemC library directory under $(SYSTEMC_HOME). Set SYSTEMC_LIB_DIR explicitly.)
endif

CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic -I$(SRC_DIR) -I. -I$(SYSTEMC_INCLUDE)
LDFLAGS ?= -L$(SYSTEMC_LIB_DIR) -Wl,-rpath,$(SYSTEMC_LIB_DIR) -lsystemc -lm

COMMON_SOURCES := $(SRC_DIR)/hdc_memory.cpp $(SRC_DIR)/hdc_accelerator.cpp $(SRC_DIR)/controller.cpp $(SRC_DIR)/foot_dataset_loader.cpp
TB_SOURCES := $(COMMON_SOURCES) $(SRC_DIR)/tb_systemc.cpp
GOLDEN_SOURCES := $(COMMON_SOURCES) $(SRC_DIR)/golden_regression.cpp
TB_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(TB_SOURCES))
GOLDEN_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(GOLDEN_SOURCES))
HEADERS := $(SRC_DIR)/systemc_types.h $(SRC_DIR)/hdc_memory.h $(SRC_DIR)/hdc_accelerator.h $(SRC_DIR)/controller.h $(SRC_DIR)/foot_dataset_loader.h $(SRC_DIR)/config_systemc.h

.PHONY: all golden clean run run-golden

all: $(TARGET)

golden: $(GOLDEN_TARGET)

$(TARGET): $(TB_OBJECTS)
	$(CXX) $(TB_OBJECTS) -o $@ $(LDFLAGS)

$(GOLDEN_TARGET): $(GOLDEN_OBJECTS)
	$(CXX) $(GOLDEN_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

run-golden: $(GOLDEN_TARGET)
	./$(GOLDEN_TARGET)

clean:
	rm -f $(TB_OBJECTS) $(GOLDEN_OBJECTS) $(TARGET) $(GOLDEN_TARGET)
