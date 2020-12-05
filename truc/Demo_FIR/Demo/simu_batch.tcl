flow package require /SCVerify
flow run /SCVerify
flow run /SCVerify/launch_make ./scverify/Verify_rtl_vhdl_msim.mk {} SIMTOOL=msim build
flow run /SCVerify/launch_make ./scverify/Verify_rtl_vhdl_msim.mk {} SIMTOOL=msim sim

