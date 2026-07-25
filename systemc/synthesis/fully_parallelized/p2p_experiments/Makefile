SYSTEMC_PREFIX ?= $(firstword $(SYSTEMC_HOME) $(SYSTEMC_ROOT) $(SYSTEMC) $(wildcard /tools.remote/systemc-2.3.2) $(wildcard /home/zora/systemc-install) $(wildcard /usr/local/systemc))
STRATUS_HOME ?= /eda/cadence/2022-23/RHELx86/STRATUS_22.01.003/share/stratus
CXX ?= g++
CXXSTD ?= c++14
TARGET ?= p2p_experiment
SRC_DIR := src
BUILD_DIR := build

SYSTEMC_INCLUDE ?= $(firstword $(wildcard $(SYSTEMC_PREFIX)/include) $(subst :, ,$(CPLUS_INCLUDE_PATH)) $(subst :, ,$(C_INCLUDE_PATH)))
SYSTEMC_LIB_DIR ?= $(firstword $(wildcard $(SYSTEMC_PREFIX)/lib-linux64) $(wildcard $(SYSTEMC_PREFIX)/lib64) $(wildcard $(SYSTEMC_PREFIX)/lib) $(subst :, ,$(LIBRARY_PATH)) $(subst :, ,$(LD_LIBRARY_PATH)))
STRATUS_INCLUDE := $(STRATUS_HOME)/include

ifeq ($(strip $(SYSTEMC_PREFIX)),)
ifeq ($(strip $(SYSTEMC_INCLUDE)),)
$(error Could not determine SystemC include path. Load a SystemC module or set SYSTEMC_HOME/SYSTEMC_ROOT/SYSTEMC/SYSTEMC_INCLUDE explicitly.)
endif
endif

ifeq ($(strip $(SYSTEMC_LIB_DIR)),)
$(error Could not determine SystemC library path. Load a SystemC module or set SYSTEMC_HOME/SYSTEMC_ROOT/SYSTEMC/SYSTEMC_LIB_DIR explicitly.)
endif

CXXFLAGS ?= -std=$(CXXSTD) -Wall -Wextra -pedantic -I$(SRC_DIR) -I$(SYSTEMC_INCLUDE) -I$(STRATUS_INCLUDE)
CXXFLAGS += $(EXTRA_CXXFLAGS)
LDFLAGS ?= -L$(SYSTEMC_LIB_DIR) -Wl,-rpath,$(SYSTEMC_LIB_DIR) -lsystemc -lm

SOURCES := $(SRC_DIR)/p2p_pipeline.cpp $(SRC_DIR)/main.cpp
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
HEADERS := $(SRC_DIR)/p2p_pipeline.h

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
	rm -rf $(BUILD_DIR) $(TARGET)
