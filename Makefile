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
