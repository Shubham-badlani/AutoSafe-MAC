# Usage (from the quartus/ folder):  quartus_sh -t run_quartus.tcl
# Compiles rtl/autosafe_mac.v for a Cyclone V device chosen automatically. No board needed.
project_new autosafe -overwrite
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE AUTO
set_global_assignment -name TOP_LEVEL_ENTITY autosafe_mac
set_global_assignment -name VERILOG_FILE ../rtl/autosafe_mac.v
set_global_assignment -name SDC_FILE autosafe.sdc
project_close
