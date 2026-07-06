SYSTEMC_HOME ?= $(firstword $(wildcard /home/zora/systemc-install) $(wildcard /usr/local/systemc))
STRATUS_HOME ?= /eda/cadence/2022-23/RHELx86/STRATUS_22.01.003/share/stratus
CXX ?= g++
TARGET ?= p2p_experiment
SRC_DIR := src
BUILD_DIR := build

SYSTEMC_INCLUDE := $(SYSTEMC_HOME)/include
SYSTEMC_LIB_DIR ?= $(firstword $(wildcard $(SYSTEMC_HOME)/lib-linux64) $(wildcard $(SYSTEMC_HOME)/lib64) $(wildcard $(SYSTEMC_HOME)/lib))
STRATUS_INCLUDE := $(STRATUS_HOME)/include

ifeq ($(strip $(SYSTEMC_LIB_DIR)),)
$(error Could not find a SystemC library directory under $(SYSTEMC_HOME). Set SYSTEMC_LIB_DIR explicitly.)
endif

CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic -I$(SRC_DIR) -I$(SYSTEMC_INCLUDE) -I$(STRATUS_INCLUDE)
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

