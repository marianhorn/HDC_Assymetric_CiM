-include   Makefile.prj

stratus:
	stratus

Makefile.prj: project.tcl
	bdw_makegen project.tcl

clean: clean_all clean_libs
	rm -rf cachelib/ bdw*/ project.tcl.user stratus.pro stratus_ide.log hls.pro
