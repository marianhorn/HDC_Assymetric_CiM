# ===================================
# CONFIG
# ===================================

CFLAGS = -O3
SRC_DIR = src
INC_DIR = include
BIN = hdc_model

D_VALUES = 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
M_VALUES = 40 50 60 70 80 90 100

RESULTS = results/results.csv

# ===================================
# DEFAULT BUILD
# ===================================

all: build

build:
	gcc $(CFLAGS) $(SRC_DIR)/*.c -I $(INC_DIR) -o $(BIN)

# ===================================
# MAIN PIPELINE
# ===================================

run: build gen_results plots

# ===================================
# GENERATE RESULTS
# ===================================

gen_results:
	@echo "D,M,Accuracy" > $(RESULTS)
	@for D in $(D_VALUES); do \
	  for M in $(M_VALUES); do \
	     echo ">>> Running D=$$D M=$$M"; \
	     python3 scripts/bitflipvector.py $$D $$M; \
	     ACC0=`./$(BIN) $$D $$M 0 | grep Accuracy | awk '{print $$2}' | tr -d '%'`; \
	     ACC1=`./$(BIN) $$D $$M 1 | grep Accuracy | awk '{print $$2}' | tr -d '%'`; \
	     echo "$$D,$$M,$$ACC0,$$ACC1" >> $(RESULTS); \
	  done; \
	done

# ===================================
# BEST RESULT
# ===================================
best:
	@echo ">>> Best parameters:"
	@python3 scripts/best_result.py $(RESULTS)

# ===================================
# PLOTTING
# ===================================

plots:
	python3 scripts/plots.py

# ===================================
# CLEAN
# ===================================

clean:
	rm -f $(BIN) $(RESULTS) plot_accuracy_vs_M.png plot_accuracy_vs_D.png
