CC = gcc
CFLAGS = -Wall -Wextra -std=c11
LDFLAGS = -lm

# Optional OpenMP support (set USE_OPENMP=1)
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

# Optional results CSV path (set RESULT_CSV_PATH=path/to/file.csv)
RESULT_CSV_PATH ?=
ifneq ($(strip $(RESULT_CSV_PATH)),)
	CFLAGS += -DRESULT_CSV_PATH=\"$(RESULT_CSV_PATH)\"
endif

# Optional config overrides (set VAR=value)
ifdef VECTOR_DIMENSION
	CFLAGS += -DVECTOR_DIMENSION=$(VECTOR_DIMENSION)
endif
ifdef NUM_LEVELS
	CFLAGS += -DNUM_LEVELS=$(NUM_LEVELS)
endif
ifdef MIN_LEVEL
	CFLAGS += -DMIN_LEVEL=$(MIN_LEVEL)
endif
ifdef MAX_LEVEL
	CFLAGS += -DMAX_LEVEL=$(MAX_LEVEL)
endif
ifdef WINDOW
	CFLAGS += -DWINDOW=$(WINDOW)
endif
ifdef N_GRAM_SIZE
	CFLAGS += -DN_GRAM_SIZE=$(N_GRAM_SIZE)
endif
ifdef ENCODER_ROLLING
	CFLAGS += -DENCODER_ROLLING=$(ENCODER_ROLLING)
endif
ifdef DOWNSAMPLE
	CFLAGS += -DDOWNSAMPLE=$(DOWNSAMPLE)
endif
ifdef NUM_CLASSES
	CFLAGS += -DNUM_CLASSES=$(NUM_CLASSES)
endif
ifdef NUM_FEATURES
	CFLAGS += -DNUM_FEATURES=$(NUM_FEATURES)
endif
ifdef NORMALIZE
	CFLAGS += -DNORMALIZE=$(NORMALIZE)
endif
ifdef CUTTING_ANGLE_THRESHOLD
	CFLAGS += -DCUTTING_ANGLE_THRESHOLD=$(CUTTING_ANGLE_THRESHOLD)
endif
ifdef PRECOMPUTED_ITEM_MEMORY
	CFLAGS += -DPRECOMPUTED_ITEM_MEMORY=$(PRECOMPUTED_ITEM_MEMORY)
endif
ifdef USE_GENETIC_ITEM_MEMORY
	CFLAGS += -DUSE_GENETIC_ITEM_MEMORY=$(USE_GENETIC_ITEM_MEMORY)
endif
ifdef OUTPUT_MODE
	CFLAGS += -DOUTPUT_MODE=$(OUTPUT_MODE)
endif
ifdef BIPOLAR_MODE
	CFLAGS += -DBIPOLAR_MODE=$(BIPOLAR_MODE)
endif
ifdef GA_DEFAULT_POPULATION_SIZE
	CFLAGS += -DGA_DEFAULT_POPULATION_SIZE=$(GA_DEFAULT_POPULATION_SIZE)
endif
ifdef GA_DEFAULT_GENERATIONS
	CFLAGS += -DGA_DEFAULT_GENERATIONS=$(GA_DEFAULT_GENERATIONS)
endif
ifdef GA_DEFAULT_CROSSOVER_RATE
	CFLAGS += -DGA_DEFAULT_CROSSOVER_RATE=$(GA_DEFAULT_CROSSOVER_RATE)
endif
ifdef GA_DEFAULT_MUTATION_RATE
	CFLAGS += -DGA_DEFAULT_MUTATION_RATE=$(GA_DEFAULT_MUTATION_RATE)
endif
ifdef GA_DEFAULT_TOURNAMENT_SIZE
	CFLAGS += -DGA_DEFAULT_TOURNAMENT_SIZE=$(GA_DEFAULT_TOURNAMENT_SIZE)
endif
ifdef GA_DEFAULT_LOG_EVERY
	CFLAGS += -DGA_DEFAULT_LOG_EVERY=$(GA_DEFAULT_LOG_EVERY)
endif
ifdef GA_DEFAULT_SEED
	CFLAGS += -DGA_DEFAULT_SEED=$(GA_DEFAULT_SEED)
endif
ifdef GA_MAX_FLIPS_CIM
	CFLAGS += -DGA_MAX_FLIPS_CIM=$(GA_MAX_FLIPS_CIM)
endif
ifdef GA_INIT_UNIFORM
	CFLAGS += -DGA_INIT_UNIFORM=$(GA_INIT_UNIFORM)
endif
ifdef GA_SELECTION_PARETO
	CFLAGS += -DGA_SELECTION_PARETO=$(GA_SELECTION_PARETO)
endif
ifdef GA_SELECTION_MULTI
	CFLAGS += -DGA_SELECTION_MULTI=$(GA_SELECTION_MULTI)
endif
ifdef GA_SELECTION_ACCURACY
	CFLAGS += -DGA_SELECTION_ACCURACY=$(GA_SELECTION_ACCURACY)
endif
ifdef GA_SELECTION_MODE
	CFLAGS += -DGA_SELECTION_MODE=$(GA_SELECTION_MODE)
endif
ifdef VALIDATION_RATIO
	CFLAGS += -DVALIDATION_RATIO=$(VALIDATION_RATIO)
endif

# Directories
SRCDIR_FOOT = foot
SRCDIR_CUSTOM = customModel
INCDIR_INFRA = hdc_infrastructure
BINDIR = build

# Source files
SRCFILES_FOOT = $(wildcard $(SRCDIR_FOOT)/*.c) $(wildcard $(INCDIR_INFRA)/*.c)
SRCFILES_CUSTOM = $(wildcard $(SRCDIR_CUSTOM)/*.c) $(wildcard $(INCDIR_INFRA)/*.c)

# Object files
OBJFILES_FOOT = $(patsubst $(SRCDIR_FOOT)/%.c,$(BINDIR)/foot_%.o,$(patsubst $(INCDIR_INFRA)/%.c,$(BINDIR)/foot_infra_%.o,$(filter-out $(SRCDIR_FOOT)/modelLS_test.c,$(SRCFILES_FOOT))))
OBJFILES_CUSTOM = $(patsubst $(SRCDIR_CUSTOM)/%.c,$(BINDIR)/custom_%.o,$(patsubst $(INCDIR_INFRA)/%.c,$(BINDIR)/custom_infra_%.o,$(filter-out $(SRCDIR_CUSTOM)/modelLS_test.c,$(SRCFILES_CUSTOM))))

# Header dependencies
DEPS_FOOT = $(wildcard $(SRCDIR_FOOT)/*.h) $(wildcard $(INCDIR_INFRA)/*.h)
DEPS_CUSTOM = $(wildcard $(SRCDIR_CUSTOM)/*.h) $(wildcard $(INCDIR_INFRA)/*.h)

# Targets
TARGET_FOOT = modelFoot
TARGET_CUSTOM = modelCustom

# Build foot EMG model
.PHONY: foot
foot: clean $(TARGET_FOOT)

$(TARGET_FOOT): $(OBJFILES_FOOT)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build custom model
.PHONY: custom
custom: clean $(TARGET_CUSTOM)

$(TARGET_CUSTOM): $(OBJFILES_CUSTOM)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Object file compilation for foot and infrastructure
$(BINDIR)/foot_%.o: $(SRCDIR_FOOT)/%.c $(DEPS_FOOT)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DFOOT_EMG -c -o $@ $<

# Object file compilation for custom model and infrastructure
$(BINDIR)/custom_%.o: $(SRCDIR_CUSTOM)/%.c $(DEPS_CUSTOM)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DCUSTOM -c -o $@ $<

# Object file compilation for shared infrastructure for foot
$(BINDIR)/foot_infra_%.o: $(INCDIR_INFRA)/%.c $(DEPS_FOOT)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DFOOT_EMG -c -o $@ $<

# Object file compilation for shared infrastructure for custom
$(BINDIR)/custom_infra_%.o: $(INCDIR_INFRA)/%.c $(DEPS_CUSTOM)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DCUSTOM -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(BINDIR)/*.o $(TARGET_FOOT) $(TARGET_CUSTOM)
