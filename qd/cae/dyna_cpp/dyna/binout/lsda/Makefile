
all: build
	.\test.exe
    
build:
	cl.exe /nologo /Zc:wchar_t /FS /Zc:strictStrings /Zc:throwingNew /Zi /MDd /GR /W3 /w34100 /w34189 /w44996 /w44456 /w44457 /w44458 /wd4577 /EHsc /DLASSO_DEBUG /DWIN64 LSDA_Wrapper.cpp btree.c lsda.c lsdaf2c.c lsdatable.c lsdatypes.c trans.c test.cpp /link /NOLOGO /DYNAMICBASE /NXCOMPAT /DEBUG /SUBSYSTEM:CONSOLE /OUT:test.exe

l2a:
	cl.exe /DWIN64 l2a.c l2amain.c btree.c lsda.c lsdaf2c.c lsdatable.c lsdatypes.c trans.c /link /OUT:l2a.exe

l2a_linux:
	g++ l2a.c l2amain.c btree.c lsda.c lsdatable.c lsdatypes.c trans.c -o l2a_linux
    
linux:
	g++ LSDA_Wrapper.cpp btree.c lsda.c lsdaf2c.c lsdatable.c lsdatypes.c trans.c test.cpp -o test_linux