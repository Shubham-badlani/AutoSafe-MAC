SIM ?= icarus
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(CURDIR)/../rtl/autosafe_mac.v
TOPLEVEL = autosafe_mac
MODULE ?= test_autosafe
BUG ?= 0
COMPILE_ARGS += -Pautosafe_mac.BUG=$(BUG)
SIM_BUILD = sim_build_bug$(BUG)
include $(shell cocotb-config --makefiles)/Makefile.sim
