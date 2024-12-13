CC = gcc
CFLAGS = -Wall -Wextra -std=c11
LDFLAGS = -lm

# RISC-V compiler
RISCV_CC = riscv32-unknown-elf-gcc
RISCV_CFLAGS = -Wall -Wextra -std=c11 -march=rv32ic

# Directories
SRCDIR_HAND = hand
SRCDIR_FOOT = foot
SRCDIR_CUSTOM = customModel
INCDIR_INFRA = hdc_infrastructure
BINDIR = build

# Source files
SRCFILES_HAND = $(wildcard $(SRCDIR_HAND)/*.c) $(wildcard $(INCDIR_INFRA)/*.c)
SRCFILES_FOOT = $(wildcard $(SRCDIR_FOOT)/*.c) $(wildcard $(INCDIR_INFRA)/*.c)
SRCFILES_CUSTOM = $(wildcard $(SRCDIR_CUSTOM)/*.c) $(wildcard $(INCDIR_INFRA)/*.c)
SRCFILES_RISCV = $(wildcard $(SRCDIR_FOOT)/*.c) $(wildcard $(INCDIR_INFRA)/*.c)

# Object files
OBJFILES_HAND = $(patsubst $(SRCDIR_HAND)/%.c,$(BINDIR)/hand_%.o,$(patsubst $(INCDIR_INFRA)/%.c,$(BINDIR)/hand_infra_%.o,$(filter-out $(SRCDIR_HAND)/modelLS_test.c,$(SRCFILES_HAND))))
OBJFILES_FOOT = $(patsubst $(SRCDIR_FOOT)/%.c,$(BINDIR)/foot_%.o,$(patsubst $(INCDIR_INFRA)/%.c,$(BINDIR)/foot_infra_%.o,$(filter-out $(SRCDIR_FOOT)/modelRISCV.c $(SRCDIR_FOOT)/modelLS_test.c,$(SRCFILES_FOOT))))
OBJFILES_CUSTOM = $(patsubst $(SRCDIR_CUSTOM)/%.c,$(BINDIR)/custom_%.o,$(patsubst $(INCDIR_INFRA)/%.c,$(BINDIR)/custom_infra_%.o,$(filter-out $(SRCDIR_CUSTOM)/modelLS_test.c,$(SRCFILES_CUSTOM))))
OBJFILES_RISCV = $(patsubst $(SRCDIR_FOOT)/%.c,$(BINDIR)/riscv_%.o,$(patsubst $(INCDIR_INFRA)/%.c,$(BINDIR)/riscv_infra_%.o,$(filter-out $(SRCDIR_FOOT)/modelFoot.c $(SRCDIR_FOOT)/modelLS_test.c,$(SRCFILES_RISCV))))

# Header dependencies
DEPS_HAND = $(wildcard $(SRCDIR_HAND)/*.h) $(wildcard $(INCDIR_INFRA)/*.h)
DEPS_FOOT = $(wildcard $(SRCDIR_FOOT)/*.h) $(wildcard $(INCDIR_INFRA)/*.h)
DEPS_CUSTOM = $(wildcard $(SRCDIR_CUSTOM)/*.h) $(wildcard $(INCDIR_INFRA)/*.h)
DEPS_RISCV = $(wildcard $(SRCDIR_FOOT)/*.h) $(wildcard $(INCDIR_INFRA)/*.h)

# Targets
TARGET_HAND = modelHand
TARGET_FOOT = modelFoot
TARGET_CUSTOM = modelCustom
TARGET_RISCV = modelRISCV

# Build hand EMG model
.PHONY: hand
hand: clean $(TARGET_HAND)

$(TARGET_HAND): $(OBJFILES_HAND)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

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

# Build RISC-V model
.PHONY: riscv
riscv: clean $(TARGET_RISCV)

$(TARGET_RISCV): $(OBJFILES_RISCV)
	$(RISCV_CC) $(RISCV_CFLAGS) -o $@ $^ $(LDFLAGS)

# Object file compilation for hand and infrastructure
$(BINDIR)/hand_%.o: $(SRCDIR_HAND)/%.c $(DEPS_HAND)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DHAND_EMG -c -o $@ $<

# Object file compilation for foot and infrastructure
$(BINDIR)/foot_%.o: $(SRCDIR_FOOT)/%.c $(DEPS_FOOT)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DFOOT_EMG -c -o $@ $<

# Object file compilation for custom model and infrastructure
$(BINDIR)/custom_%.o: $(SRCDIR_CUSTOM)/%.c $(DEPS_CUSTOM)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DCUSTOM -c -o $@ $<

# Object file compilation for shared infrastructure for hand
$(BINDIR)/hand_infra_%.o: $(INCDIR_INFRA)/%.c $(DEPS_HAND)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DHAND_EMG -c -o $@ $<

# Object file compilation for shared infrastructure for foot
$(BINDIR)/foot_infra_%.o: $(INCDIR_INFRA)/%.c $(DEPS_FOOT)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DFOOT_EMG -c -o $@ $<

# Object file compilation for shared infrastructure for custom
$(BINDIR)/custom_infra_%.o: $(INCDIR_INFRA)/%.c $(DEPS_CUSTOM)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DCUSTOM -c -o $@ $<

# Object file compilation for RISC-V
$(BINDIR)/riscv_%.o: $(SRCDIR_FOOT)/%.c $(DEPS_RISCV)
	@mkdir -p $(BINDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c -o $@ $<

# Object file compilation for RISC-V infrastructure
$(BINDIR)/riscv_infra_%.o: $(INCDIR_INFRA)/%.c $(DEPS_RISCV)
	@mkdir -p $(BINDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(BINDIR)/*.o $(TARGET_HAND) $(TARGET_FOOT) $(TARGET_CUSTOM) $(TARGET_RISCV)
