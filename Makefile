CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3 -march=native -mtune=native -flto -DNDEBUG
LDFLAGS = -lm -flto

# Optional OpenMP support:
#   USE_OPENMP=1    force enable
#   USE_OPENMP=0    force disable
#   USE_OPENMP=auto enable when compiler supports -fopenmp (default)
USE_OPENMP ?= auto
OPENMP_SUPPORTED := $(shell printf "int main(void){return 0;}\n" | $(CC) -x c -fopenmp -fsyntax-only - >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(USE_OPENMP),1)
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
else ifeq ($(USE_OPENMP),auto)
ifeq ($(OPENMP_SUPPORTED),1)
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif
endif

ifdef ITEM_MEM_SEED
	CFLAGS += -DITEM_MEM_SEED=$(ITEM_MEM_SEED)
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
ifdef N_GRAM_SIZE
	CFLAGS += -DN_GRAM_SIZE=$(N_GRAM_SIZE)
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
ifdef OUTPUT_MODE
	CFLAGS += -DOUTPUT_MODE=$(OUTPUT_MODE)
endif
ifdef VALIDATION_RATIO
	CFLAGS += -DVALIDATION_RATIO=$(VALIDATION_RATIO)
endif

# Directories
SRCDIR_FOOT = foot
INCDIR_INFRA = hdc_infrastructure
BINDIR = build

# Source files
SRCFILES_FOOT = $(wildcard $(SRCDIR_FOOT)/*.c) $(wildcard $(INCDIR_INFRA)/*.c)

# Object files
OBJFILES_FOOT = $(patsubst $(SRCDIR_FOOT)/%.c,$(BINDIR)/foot_%.o,$(patsubst $(INCDIR_INFRA)/%.c,$(BINDIR)/foot_infra_%.o,$(filter-out $(SRCDIR_FOOT)/modelLS_test.c,$(SRCFILES_FOOT))))

# Header dependencies
DEPS_FOOT = $(wildcard $(SRCDIR_FOOT)/*.h) $(wildcard $(INCDIR_INFRA)/*.h)

# Targets
TARGET_FOOT = modelFoot

# Build foot EMG model
.PHONY: foot
foot: clean $(TARGET_FOOT)

$(TARGET_FOOT): $(OBJFILES_FOOT)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Object file compilation for foot and infrastructure
$(BINDIR)/foot_%.o: $(SRCDIR_FOOT)/%.c $(DEPS_FOOT)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DFOOT_EMG -c -o $@ $<

# Object file compilation for shared infrastructure for foot
$(BINDIR)/foot_infra_%.o: $(INCDIR_INFRA)/%.c $(DEPS_FOOT)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DFOOT_EMG -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(BINDIR)/*.o $(TARGET_FOOT)
