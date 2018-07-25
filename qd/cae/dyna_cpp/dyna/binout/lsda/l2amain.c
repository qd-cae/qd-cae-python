#include <ctype.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "l2a_version.h"
#include "lsda.h"

extern int
translate_secforc(int handle);
extern int
translate_rwforc(int handle);
extern int
translate_nodout(int handle);
extern int
translate_curvout(int handle);
extern int
translate_nodouthf(int handle);
extern int
translate_elout(int handle);
extern int
translate_eloutdet(int handle);
extern int
translate_glstat(int handle);
extern int
translate_ssstat(int handle);
extern int
translate_deforc(int handle);
extern int
translate_matsum(int handle);
extern int
translate_ncforc(int handle);
extern int
translate_rcforc(int handle);
extern int
translate_spcforc(int handle);
extern int
translate_swforc(int handle);
extern int
translate_abstat(int handle);
extern int
translate_abstat_cpm(int handle);
extern int
translate_abstat_pbm(int handle);
extern int
translate_cpm_sensor(int handle);
extern int
translate_pgstat(int handle);
extern int
translate_pg_sensor(int handle);
extern int
translate_nodfor(int handle);
extern int
translate_bndout(int handle);
extern int
translate_rbdout(int handle);
extern int
translate_gceout(int handle);
extern int
translate_sleout(int handle);
extern int
translate_sbtout(int handle);
extern int
translate_jntforc(int handle);
extern int
translate_sphout(int handle);
extern int
translate_defgeo(int handle);
extern int
translate_dcfail(int handle);
extern int
translate_tprint(int handle);
extern int
translate_trhist(int handle);
extern int
translate_dbsensor(int handle);
extern int
translate_dbfsi(int handle);
extern int
translate_elout_ssd(int handle);
extern int
translate_elout_spcm(int handle);
extern int
translate_elout_psd(int handle);
extern int
translate_nodout_ssd(int handle);
extern int
translate_nodout_spcm(int handle);
extern int
translate_nodout_psd(int handle);
extern int
translate_pllyout(int handle);
extern int
translate_dem_rcforc(int handle);
extern int
translate_disbout(int handle);
extern int
translate_dem_trhist(int handle);
extern int
translate_traleh(int handle);
extern void
l2a_set_output_path(char* pwd);

typedef struct _ent
{
  char name[64];
  int (*trans)(int);
  int active;
} ENTRY;

#define NUM_ENTRIES 46

ENTRY list[NUM_ENTRIES] = { { "secforc", translate_secforc, 0 },
                            { "rwforc", translate_rwforc, 0 },
                            { "curvout", translate_curvout, 0 },
                            { "nodout", translate_nodout, 0 },
                            { "nodouthf", translate_nodouthf, 0 },
                            { "elout", translate_elout, 0 },
                            { "eloutdet", translate_eloutdet, 0 },
                            { "glstat", translate_glstat, 0 },
                            { "ssstat", translate_ssstat, 0 },
                            { "deforc", translate_deforc, 0 },
                            { "matsum", translate_matsum, 0 },
                            { "ncforc", translate_ncforc, 0 },
                            { "rcforc", translate_rcforc, 0 },
                            { "spcforc", translate_spcforc, 0 },
                            { "swforc", translate_swforc, 0 },
                            { "abstat", translate_abstat, 0 },
                            { "abstat_cpm", translate_abstat_cpm, 0 },
                            { "abstat_pbm", translate_abstat_pbm, 0 },
                            { "cpm_sensor", translate_cpm_sensor, 0 },
                            { "pg_stat", translate_pgstat, 0 },
                            { "pg_sensor", translate_pg_sensor, 0 },
                            { "nodfor", translate_nodfor, 0 },
                            { "bndout", translate_bndout, 0 },
                            { "rbdout", translate_rbdout, 0 },
                            { "gceout", translate_gceout, 0 },
                            { "sleout", translate_sleout, 0 },
                            { "sbtout", translate_sbtout, 0 },
                            { "jntforc", translate_jntforc, 0 },
                            { "sphout", translate_sphout, 0 },
                            { "defgeo", translate_defgeo, 0 },
                            { "dcfail", translate_dcfail, 0 },
                            { "tprint", translate_tprint, 0 },
                            { "trhist", translate_trhist, 0 },
                            { "dbsensor", translate_dbsensor, 0 },
                            { "dbfsi", translate_dbfsi, 0 },
                            { "elout_ssd", translate_elout_ssd, 0 },
                            { "elout_spcm", translate_elout_spcm, 0 },
                            { "elout_psd", translate_elout_psd, 0 },
                            { "nodout_ssd", translate_nodout_ssd, 0 },
                            { "nodout_spcm", translate_nodout_spcm, 0 },
                            { "nodout_psd", translate_nodout_psd, 0 },
                            { "pllyout", translate_pllyout, 0 },
                            { "dem_rcforc", translate_dem_rcforc, 0 },
                            { "disbout", translate_disbout, 0 },
                            { "demtrh", translate_dem_trhist, 0 },
                            { "traleh", translate_traleh, 0 } };

void
print_help(char* pname)
{
  printf("\n\nSyntax: %s -v -h -j INFILE [INFILE]....\n", pname);
  printf("        -v = print version information and continue\n");
  printf("        -h = print this help text\n");
  printf("        -j = use job ids:\n");
  printf(
    "             the first input file is checked for a JOBID type prefix\n");
  printf("             and if found, this prefex is prepended to all output "
         "files\n\n");
}

static int
mycmp(const void* v1, const void* v2)
{
  char* s1 = *(char**)v1;
  char* s2 = *(char**)v2;
  return strcmp(s1, s2);
}

int
main(int argc, char* argv[])
{
  int handle, i, j;
  int typeid, filenum;
  LSDA_Length length;
  LSDADir* dp;
  char name[32];
  int target = 0;
  int nopen, first, last;
  char* toopen[1024];
#ifdef MPPWIN
  int optind = 1;
#else
  extern int optind, opterr, optopt;
#endif
  int use_jobid;
  char outpath[128];

  if (argc < 2) {
    print_help(argv[0]);
    exit(1);
  }

  use_jobid = 0;
#ifdef MPPWIN
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h")) {
      print_help(argv[0]);
      exit(3);
    }
    if (!strcmp(argv[i], "-v")) {
      printf("\nLSTC l2a Utility Version %s\n\n", L2A_REVISION);
    }
    if (!strcmp(argv[i], "-j")) {
      use_jobid = 1;
    }
  }
#else
  opterr = 0;
  for (;;) {
    i = getopt(argc, argv, "hvj");
    if (i == -1)
      break;
    if (i == '?') {
      printf("Unrecognized option %c\n", optopt);
      print_help(argv[0]);
      exit(2);
    }
    if (i == 'h') {
      print_help(argv[0]);
      exit(3);
    }
    if (i == 'v') {
      printf("\nLSTC l2a Utility Version %s\n\n", L2A_REVISION);
    }
    if (i == 'j') {
      use_jobid = 1;
    }
  }
#endif
  /*
    Go through all the command line args and pull out any that
    match ENTRY names.  Flag those as the only ones we will translate.
  */
  nopen = 0;
  for (i = optind; i < argc; i++) {
#ifdef MPPWIN
    if (!strcmp(argv[i], "-h"))
      continue;
    if (!strcmp(argv[i], "-j"))
      continue;
    if (!strcmp(argv[i], "-v"))
      continue;
#endif
    for (j = 0; j < NUM_ENTRIES; j++) {
      if (strcmp(argv[i], list[j].name) == 0) {
        target = 1;
        list[j].active = 1;
        break;
      }
    }
    /*  Must be a file name.  Add to list of files to open */
    if (j == NUM_ENTRIES)
      toopen[nopen++] = argv[i];
  }
  if (nopen == 0)
    exit(0);

  /* sort the "toopen" list so any files of the same jobid are together */
  qsort(toopen, nopen, sizeof(char*), mycmp);
  toopen[nopen] = 0;
  first = 0;

  while (toopen[first]) {
    strcpy(outpath, "./");
    last = nopen - 1;
    if (use_jobid) {
      int len, i;
      char* cp;
      /* extract the jobid from the first target file */
      strcat(outpath, toopen[first]);
      cp = strchr(outpath + 2, '.');
      if (cp) {
        cp[1] = 0; /* keep trailing . */
        len = cp - outpath - 1;
        for (i = first; toopen[i]; i++)
          if (strncmp(outpath + 2, toopen[i], len) == 0)
            last = i;
      } else {
        outpath[2] = 0;
        for (i = first; toopen[i]; i++)
          if (strchr(toopen[i], '.') == NULL)
            last = i;
      }
    }

    l2a_set_output_path(outpath);

    handle = lsda_open_many(toopen + first, last - first + 1);
    if (handle < 0) {
      printf("One of the arguments does not appear to be a valid LSDA file\n");
      exit(1);
    }
    dp = lsda_opendir(handle, "/");

    for (;;) {
      lsda_readdir(dp, name, &typeid, &length, &filenum);
      if (name[0] == 0)
        break; /* end of listing */
      for (i = 0; i < NUM_ENTRIES; i++) {
        if (strcmp(name, list[i].name) == 0 && list[i].active == target) {
          list[i].trans(handle);
          break;
        }
      }
    }
    lsda_close(handle);
    first = last + 1;
  }
  exit(0);
}
