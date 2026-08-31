SYSTEMC_PREFIX ?= $(firstword $(SYSTEMC_HOME) $(SYSTEMC_ROOT) $(SYSTEMC) $(wildcard /tools.remote/systemc-2.3.2) $(wildcard /home/zora/systemc-install) $(wildcard /usr/local/systemc))
CXX ?= g++
CXXSTD ?= c++17
TARGET ?= main
GOLDEN_TARGET ?= systemc_hdc_golden_regression
RTL_TRACE_TARGET ?= systemc_hdc_rtl_trace_export
RTL_REPLAY_TARGET ?= systemc_hdc_rtl_trace_replay
LEGACY_TARGETS := systemc_hdc_tb
SRC_DIR := src
BUILD_DIR := build

SYSTEMC_INCLUDE ?= $(firstword $(wildcard $(SYSTEMC_PREFIX)/include) $(subst :, ,$(CPLUS_INCLUDE_PATH)) $(subst :, ,$(C_INCLUDE_PATH)))
SYSTEMC_LIB_DIR ?= $(firstword $(wildcard $(SYSTEMC_PREFIX)/lib-linux64) $(wildcard $(SYSTEMC_PREFIX)/lib64) $(wildcard $(SYSTEMC_PREFIX)/lib) $(subst :, ,$(LIBRARY_PATH)) $(subst :, ,$(LD_LIBRARY_PATH)))

ifeq ($(strip $(SYSTEMC_PREFIX)),)
ifeq ($(strip $(SYSTEMC_INCLUDE)),)
$(error Could not determine SystemC include path. Load a SystemC module or set SYSTEMC_HOME/SYSTEMC_ROOT/SYSTEMC/SYSTEMC_INCLUDE explicitly.)
endif
endif

ifeq ($(strip $(SYSTEMC_LIB_DIR)),)
$(error Could not determine SystemC library path. Load a SystemC module or set SYSTEMC_HOME/SYSTEMC_ROOT/SYSTEMC/SYSTEMC_LIB_DIR explicitly.)
endif

CXXFLAGS ?= -std=$(CXXSTD) -Wall -Wextra -pedantic -I$(SRC_DIR) -I. -I$(SYSTEMC_INCLUDE)
LDFLAGS ?= -L$(SYSTEMC_LIB_DIR) -Wl,-rpath,$(SYSTEMC_LIB_DIR) -lsystemc -lm

COMMON_SOURCES := $(SRC_DIR)/hdc_memory.cpp $(SRC_DIR)/hdc_accelerator.cpp $(SRC_DIR)/controller.cpp $(SRC_DIR)/foot_dataset_loader.cpp
TB_SOURCES := $(COMMON_SOURCES) $(SRC_DIR)/main.cpp
GOLDEN_SOURCES := $(COMMON_SOURCES) $(SRC_DIR)/golden_regression.cpp
RTL_TRACE_SOURCES := $(COMMON_SOURCES) $(SRC_DIR)/rtl_trace_export.cpp
RTL_REPLAY_SOURCES := $(COMMON_SOURCES) $(SRC_DIR)/rtl_trace_replay.cpp
TB_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(TB_SOURCES))
GOLDEN_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(GOLDEN_SOURCES))
RTL_TRACE_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(RTL_TRACE_SOURCES))
RTL_REPLAY_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(RTL_REPLAY_SOURCES))
HEADERS := $(SRC_DIR)/systemc_types.h $(SRC_DIR)/hdc_transactions.h $(SRC_DIR)/hdc_memory.h $(SRC_DIR)/hdc_accelerator.h $(SRC_DIR)/controller.h $(SRC_DIR)/foot_dataset_loader.h $(SRC_DIR)/config_systemc.h $(SRC_DIR)/generated_cim_rom_dataset00_banked.h

.PHONY: all golden rtl-trace rtl-replay clean run run-golden

all: $(TARGET)

golden: $(GOLDEN_TARGET)

rtl-trace: $(RTL_TRACE_TARGET)

rtl-replay: $(RTL_REPLAY_TARGET)

$(TARGET): $(TB_OBJECTS)
	$(CXX) $(TB_OBJECTS) -o $@ $(LDFLAGS)

$(GOLDEN_TARGET): $(GOLDEN_OBJECTS)
	$(CXX) $(GOLDEN_OBJECTS) -o $@ $(LDFLAGS)

$(RTL_TRACE_TARGET): $(RTL_TRACE_OBJECTS)
	$(CXX) $(RTL_TRACE_OBJECTS) -o $@ $(LDFLAGS)

$(RTL_REPLAY_TARGET): $(RTL_REPLAY_OBJECTS)
	$(CXX) $(RTL_REPLAY_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

run-golden: $(GOLDEN_TARGET)
	./$(GOLDEN_TARGET)

clean:
	rm -f $(TB_OBJECTS) $(GOLDEN_OBJECTS) $(RTL_TRACE_OBJECTS) $(RTL_REPLAY_OBJECTS) $(TARGET) $(GOLDEN_TARGET) $(RTL_TRACE_TARGET) $(RTL_REPLAY_TARGET) $(LEGACY_TARGETS)
