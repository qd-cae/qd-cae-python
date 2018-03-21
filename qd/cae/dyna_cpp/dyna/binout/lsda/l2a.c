/*
  Copyright (C) 2002
  by Livermore Software Technology Corp. (LSTC)
  All rights reserved
*/
#include <stdlib.h>
#include <stdio.h>
#ifdef MACX
#include <sys/malloc.h>
#else
#include <malloc.h>
#endif
#include <ctype.h>
#include <string.h>
#include <math.h>

/* Requested by ARUP for portability */
#ifndef LOCALIZED
#include "lsda.h"
#endif

/*
** structure definitions
*/
typedef struct {
  char states[20][32];
  int idsize;
  int *ids,*mat,*state,*locats,*locatn,*nip,*nqt;
  float *sxx,*syy,*szz,*sxy,*syz,*szx,*yield,*effsg;
} MDSOLID;
typedef struct {
  char states[20][32];
  int idsize,nhv;
  int *ids,*mat,*ndata,*nhist;
  float *data,*hist,*strain;
} MDHIST;  /* for solid_hist, shell_hist, etc */
typedef struct {
  char states[20][32];
  char system[8];
  int *state,*ids,*mat,*nip,*npl,*locats,*locatn,*damage;
  float *sxx,*syy,*szz,*sxy,*syz,*szx;
  float *exx,*eyy,*ezz,*exy,*eyz,*ezx;
  float *lxx,*lyy,*lzz,*lxy,*lyz,*lzx;
  float *uxx,*uyy,*uzz,*uxy,*uyz,*uzx;
  float *yield,*effsg;
} MDTSHELL;
typedef struct {
  int idsize, dsize;
  int *ids, *mat, *nip, *mtype;
  float *axial, *shears, *sheart, *moments, *momentt, *torsion;
  float *clength, *vforce;
  float *s11,*s12,*s31,*plastic;
} MDBEAM;
typedef struct {
  char states[20][32];
  char system[8];
  int idsize, dsize;
  int *ids, *mat, *nip, *state, *iop, *npl, *nqt, *locats, *locatn;
  int *damage;
  float *sxx, *syy, *szz;
  float *sxy, *syz, *szx;
  float *ps;
  float *exx, *eyy, *ezz;
  float *exy, *eyz, *ezx;
  float *lxx, *lyy, *lzz;
  float *lxy, *lyz, *lzx;
  float *uxx, *uyy, *uzz;
  float *uxy, *uyz, *uzx;
} MDSHELL;
typedef struct {
  char states[20][32];
  char system[8];
  int *ids;
  float *factor, *lyield, *uyield;
  float *lxx, *lyy, *lzz,*lxy, *lyz, *lzx;
  float *uxx, *uyy, *uzz,*uxy, *uyz, *uzx;
} MDNODAVG;
typedef struct {
  int num;
  int *ids,*setid,*locats,*locatn;
  float *xf,*yf,*zf,*e,*xm,*ym,*zm;
} BND_DATA;

/*
** Function prototypes
*/
int translate_secforc(int handle);
int translate_rwforc(int handle);
int translate_nodout(int handle);
int translate_curvout(int handle);
int translate_nodouthf(int handle);
int translate_elout(int handle);
int translate_eloutdet(int handle);
int translate_glstat(int handle);
int translate_ssstat(int handle);
int translate_deforc(int handle);
int translate_matsum(int handle);
int translate_ncforc(int handle);
int translate_rcforc(int handle);
int translate_spcforc(int handle);
int translate_swforc(int handle);
int translate_abstat(int handle);
int translate_abstat_cpm(int handle);
int translate_abstat_pbm(int handle);
int translate_cpm_sensor(int handle);
int translate_cpm_sensor_new(int handle);
int translate_pgstat(int handle);
int translate_pg_sensor(int handle);
int translate_pg_sensor_new(int handle);
int translate_nodfor(int handle);
int translate_bndout(int handle);
int translate_rbdout(int handle);
int translate_gceout(int handle);
int translate_sleout(int handle);
int translate_sbtout(int handle);
int translate_jntforc(int handle);
int translate_sphout(int handle);
int translate_defgeo(int handle);
int translate_dcfail(int handle);
int translate_tprint(int handle);
int translate_trhist(int handle);
int translate_dbsensor(int handle);
int translate_dbfsi(int handle);
int translate_elout_ssd(int handle);
int translate_elout_spcm(int handle);
int translate_elout_psd(int handle);
int translate_nodout_ssd(int handle);
int translate_nodout_spcm(int handle);
int translate_nodout_psd(int handle);
int translate_pllyout(int handle);
int translate_dem_rcforc(int handle);
int translate_disbout(int handle);
int translate_dem_trhist(int handle);

void output_title(int , char *, FILE *);
void output_legend(int, FILE *, int, int);
int elout_solid(FILE *, int, int, MDSOLID *);
int elout_tshell(FILE *, int, int, MDTSHELL *);
int elout_beam(FILE *, int, int, MDBEAM *);
int elout_shell(FILE *, int, int, MDSHELL *);
int eloutdet_solid(FILE *, int, int, int, int, int, int, MDSOLID *);
int eloutdet_tshell(FILE *, int, int, int, int, int, int,MDTSHELL *);
int eloutdet_shell(FILE *, int, int, int, int, int, int,MDSHELL *);
int eloutdet_nodavg(FILE *, int, int, MDNODAVG *);
int comp_ncn(const void *, const void *);
int bndout_dn(FILE *, int, int, BND_DATA *, float *, float *, float *, float *, int *);
int bndout_dr(FILE *, int, int, BND_DATA *, float *, float *, float *, float *, int *);
int bndout_p(FILE *, int, int, BND_DATA *, int *);
int bndout_vn(FILE *, int, int, BND_DATA *, float *, float *, float *, float *, int *);
int bndout_vr(FILE *, int, int, BND_DATA *, float *, float *, float *, float *, int *);
int bndout_or(FILE *, int, int, BND_DATA *, float *, float *, float *, float *, int *);

/*
** New static functions
*/
static char *tochar2(float,int);
static void db_floating_format(float, int, char *, int, int);

static char output_path[256];
static char output_file[256];
/*
 * Requested by ARUP for portability:
 */
#ifdef LOCALIZED
#include "lsda_localizations.inc"
#else
static void write_message(FILE *fp, char *string)
{
  if (!fp)
    printf("Unable to open : %s\n",string);
  else
    printf("Writing : %s\n",string);
}
#endif

void l2a_set_output_path(char *pwd)
{
  strcpy(output_path, pwd);
}

void output_legend(int handle, FILE *fp,int NO_LONGER_USED, int last)
{
  int i,bpt,typid,filenum,*ids,num;
  char *legend;
  LSDA_Length length;
  char format[64];
  static int need_begin = 1;

  lsda_queryvar(handle,"legend_ids",&typid,&length,&filenum);
  num = length;
  if(num > 0) {
/*
  The length of each title is either 70 or 80 (or maybe changed in future?)
  So figure out what it is, and create the proper output format string
  bpt = bytes per title
*/
    lsda_queryvar(handle,"legend",&typid,&length,&filenum);
    bpt = length / num;
    sprintf(format,"%%9d     %%.%ds\n",bpt);
    ids = (int *) malloc(num*sizeof(int));
    legend = (char *) malloc(bpt*num);
    lsda_read(handle,LSDA_INT,"legend_ids",0,num,ids);
    lsda_read(handle,LSDA_I1,"legend",0,bpt*num,legend);
    if(need_begin) {
      fprintf(fp,"\n{BEGIN LEGEND}\n");
      fprintf(fp," Entity #        Title\n");
      need_begin=0;
    }
    for(i=0; i<num; i++) {
      fprintf(fp,format,ids[i],legend+bpt*i);
    }
    free(legend);
    free(ids);
  }
  if(last) {   /* have to finish off legend.... */
    if(need_begin)
      fprintf(fp,"\n\n");  /* legend never started -- leave empty */
    else
      fprintf(fp,"\n{END LEGEND}\n");  /* finish legend */
    need_begin = 1;
  }
  return;
}
void output_legend_nosort(int handle, FILE *fp,int NO_LONGER_USED, int last,
                         int *no_sort)
{
  int i,bpt,typid,filenum,*ids,num,k;
  char *legend;
  LSDA_Length length;
  char format[64];
  static int need_begin = 1;

  lsda_queryvar(handle,"legend_ids",&typid,&length,&filenum);
  num = length;
  if(num > 0) {
/*
  The length of each title is either 70 or 80 (or maybe changed in future?)
  So figure out what it is, and create the proper output format string
  bpt = bytes per title
*/
    lsda_queryvar(handle,"legend",&typid,&length,&filenum);
    bpt = length / num;
    sprintf(format,"%%9d     %%.%ds\n",bpt);
    ids = (int *) malloc(num*sizeof(int));
    legend = (char *) malloc(bpt*num);
    lsda_read(handle,LSDA_INT,"legend_ids",0,num,ids);
    lsda_read(handle,LSDA_I1,"legend",0,bpt*num,legend);
    if(need_begin) {
      fprintf(fp,"\n{BEGIN LEGEND}\n");
      fprintf(fp," Entity #        Title\n");
      need_begin=0;
    }
    for(k=0; k<num; k++) {
      i=no_sort[k]-1;
      fprintf(fp,format,ids[i],legend+bpt*i);
    }
    if(last) fprintf(fp,"{END LEGEND}\n\n");
    free(legend);
    free(ids);
  }
  if(last) {   /* have to finish off legend.... */
    if(need_begin)
      fprintf(fp,"\n\n");  /* legend never started -- leave empty */
    else
      fprintf(fp,"\n{END LEGEND}\n");  /* finish legend */
    need_begin = 0;
  }
  return;
}
void output_title(int handle, char *path, FILE *fp)
{
  int typid,filenum;
  char pwd[512];
  char title[81],version[13],revision[11],date[11];
  char s1[32],s2[32],sout[64];
  int i1,i2,major,minor;
  LSDA_Length length;

  /* save current location in file so we can restore it */
  strcpy(pwd,lsda_getpwd(handle));

  /* move to metadata directory where stuff should live */
  lsda_cd(handle,path);

  lsda_read(handle,LSDA_I1,"title",0,80,title);
  title[72]=0;
  lsda_read(handle,LSDA_I1,"date",0,10,date);
  date[10]=0;
/*
  write header
*/
  fprintf(fp," %s\n",title);
/*
  Older version of this file had "version" as 10 bytes, and did
  not have "revision" at all.  When "revision" was added, "version"
  became 12 bytes, and the output format changed a bit...
*/

  lsda_queryvar(handle,"revision",&typid,&length,&filenum);
  if(typid > 0) {  /* revision exists -- new format */
    lsda_read(handle,LSDA_I1,"version",0,12,version);
    version[12]=0;
    lsda_read(handle,LSDA_I1,"revision",0,10,revision);
    revision[10]=0;
    /*
      version is something like "ls971 beta" and revision is
      something like " 5434.012 ".  Either might have leading and/or
      trailing spaces.  I want to put together a string like:
      "ls971.5434.012 beta" (the "beta" is optional)
    */
    i1=sscanf(version,"%s %s",s1,s2);
    if(i1 < 2) s2[0]=0;
    i2=sscanf(revision,"%d.%d",&major,&minor);
    if(i2 < 2) {
      sprintf(sout,"%s.%d %s",s1,major,s2);
    } else {
      sprintf(sout,"%s.%d.%3.3d %s",s1,major,minor,s2);
    }
    fprintf(fp,"                         ls-dyna %-22s date %s\n",sout,date);
  } else {          /* no revision -- old format */
    lsda_read(handle,LSDA_I1,"version",0,10,version);
    version[10]=0;
    fprintf(fp,"                         ls-dyna (version %s)     date %s\n",version,date);
  }
  lsda_cd(handle,pwd);
}
LSDADir *next_dir(int handle, char *pwd, LSDADir *dp, char *name)
/*
  Look for data directories.  The first time this is called
  (dp==NULL), we open the directory and start reading the listing.
  We return each time we find a subdirectory in the form "dXXXXXXX"
  where X are all digits.  readdir returns things in alphabetic
  order, so we should be getting them in the correct order, no problem.
  When we can't find any more, return NULL.
  We fill in the directory name in "name", and CD in to the new
  directory.  And we return the new value of dp -- this is just
  easier than passing &dp and having all those extra * floating around...
*/
{
  char path[64];
  int typid;
  LSDA_Length length;
  int i,filenum;

  if(!dp) dp = lsda_opendir(handle,pwd);
  if(!dp) return NULL;
  while (1) {
    lsda_readdir(dp,name,&typid,&length,&filenum);

    if(name[0]==0) {  /* end of directory listing */
      lsda_closedir(dp);
      return NULL;
    }
    if(typid==0 && name[0] == 'd') {
      for(i=1; name[i] && isdigit(name[i]); i++)
        ;
      if(!name[i]) {
        sprintf(path,"%s/%s",pwd,name);
        lsda_cd(handle,path);
        return dp;
      }
    }
  }
}

int next_dir_6or8digitd(int handle, char *pwd, int state)
{
  char dirname[128];
  int typid;
  LSDA_Length length;
  int i,filenum;

  if(state<=999999) 
    sprintf(dirname,"%s/d%6.6d",pwd,state);
  else
    sprintf(dirname,"%s/d%8.8d",pwd,state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  return 1;
}


/*
  SECFORC file
*/
int translate_secforc(int handle)
{
  int i,typid,num,filenum,state,need_renumber;
  LSDA_Length length;
  char dirname[32];
  int *ids, *idfix;
  int *rigidbody;
  int *accelerometer;
  int *coordinate_system;
  float time;
  float *x_force;
  float *y_force;
  float *z_force;
  float *total_force;
  float *x_moment;
  float *y_moment;
  float *z_moment;
  float *total_moment;
  float *x_centroid;
  float *y_centroid;
  float *z_centroid;
  float *area;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/secforc/metadata") == -1) return 0;
  printf("Extracting SECFORC data\n");
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num=length;

/*
  allocate memory to read in 1 state
*/
  ids = (int *) malloc(num*sizeof(int));
  rigidbody = (int *) malloc(num*sizeof(int));
  accelerometer = (int *) malloc(num*sizeof(int));
  coordinate_system = (int *) malloc(num*sizeof(int));
  x_force = (float *) malloc(num*sizeof(float));
  y_force = (float *) malloc(num*sizeof(float));
  z_force = (float *) malloc(num*sizeof(float));
  total_force = (float *) malloc(num*sizeof(float));
  x_moment = (float *) malloc(num*sizeof(float));
  y_moment = (float *) malloc(num*sizeof(float));
  z_moment = (float *) malloc(num*sizeof(float));
  total_moment = (float *) malloc(num*sizeof(float));
  x_centroid = (float *) malloc(num*sizeof(float));
  y_centroid = (float *) malloc(num*sizeof(float));
  z_centroid = (float *) malloc(num*sizeof(float));
  area = (float *) malloc(num*sizeof(float));
/*
  Read metadata
*/
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
  /*
   * There is a bug in some of the earlier releases of DYNA, whereby the
   * MPP code assigns the wrong ids to the cross sections: each processor
   * numbers their own from 1-n locally.  The result is that the id numbers
   * actually written can be wrong.  The actual data is correct, and in the
   * correct order.  So, if we can detect this condition, we will just
   * renumber them here from 1 to num.  In the broken code, each section
   * gets, as its id, the max of its ids on all the processors it lives on.
   * The only thing we can reliably look for is collisions.  If there are
   * no collisions, there is no why to be SURE the data is not correct.
   * But watch for ids > num, in case the user assigned IDs (which are
   * honored, even in the broken code).
   */
  idfix = rigidbody;   /* use this as scratch space. */
  memset(idfix,0,num*sizeof(int));
  need_renumber = 0;
  for(i=0; i<num; i++) {
    if(ids[i] > 0 && ids[i] <= num) {
      if(++idfix[ids[i]-1] > 1)
        need_renumber=1;
    }
  }
  if(need_renumber) {
    for(i=0; i<num; i++) {
      if(ids[i] > 0 && ids[i] <= num)  /* leave user given IDS alone... */
        ids[i]=i+1;
    }
  }
  lsda_read(handle,LSDA_INT,"rigidbody",0,num,rigidbody);
  lsda_read(handle,LSDA_INT,"accelerometer",0,num,accelerometer);
  lsda_queryvar(handle,"coordinate_system",&typid,&length,&filenum);
  if(typid > 0) {
    lsda_read(handle,LSDA_INT,"coordinate_system",0,num,coordinate_system);
  } else {
    memset(coordinate_system,0,num*sizeof(int));
  }
/*
  open file and write header
*/
  sprintf(output_file,"%ssecforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/secforc/metadata",fp);
  output_legend(handle,fp,1,1);
  fprintf(fp,"\n\n");
  fprintf(fp," line#1  section#     time        x-force     y-force     z-force    magnitude\n");
  fprintf(fp," line#2  resultant  moments       x-moment    y-moment    z-moment   magnitude\n");
  fprintf(fp," line#3  centroids                x           y           z            area  \n");

/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/secforc",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_force",0,num,x_force) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_force",0,num,y_force) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_force",0,num,z_force) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_force",0,num,total_force) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_moment",0,num,x_moment) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_moment",0,num,y_moment) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_moment",0,num,z_moment) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_moment",0,num,total_moment) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_centroid",0,num,x_centroid) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_centroid",0,num,y_centroid) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_centroid",0,num,z_centroid) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"area",0,num,area) != num) break;
    for(i=0; i<num; i++) {
      fprintf(fp,"%12d%15.5E",ids[i],time);
      fprintf(fp,"%15.4E%12.4E%12.4E%12.4E\n",x_force[i],y_force[i],z_force[i],total_force[i]);
      if(rigidbody[i] != 0) {
        fprintf(fp,"rb ID =%8d            ",rigidbody[i]);
      } else if(accelerometer[i] != 0) {
        fprintf(fp,"ac ID =%8d            ",accelerometer[i]);
      } else if(coordinate_system[i] != 0) {
        fprintf(fp,"cs ID =%8d            ",coordinate_system[i]);
      } else {
        fprintf(fp,"global system              ");
      }
      fprintf(fp,"%15.4E%12.4E%12.4E%12.4E\n",x_moment[i],y_moment[i],z_moment[i],total_moment[i]);
      fprintf(fp,"                           ");
      fprintf(fp,"%15.4E%12.4E%12.4E%12.4E\n\n",x_centroid[i],y_centroid[i],z_centroid[i],area[i]);
    }
  }
  fclose(fp);
  free(area);
  free(z_centroid);
  free(y_centroid);
  free(x_centroid);
  free(total_moment);
  free(z_moment);
  free(y_moment);
  free(x_moment);
  free(total_force);
  free(z_force);
  free(y_force);
  free(x_force);
  free(coordinate_system);
  free(accelerometer);
  free(rigidbody);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  RWFORC file
*/
typedef struct _ipair {
  int id;
  int pos;
} IPAIR;

int ipsort(const void *v1, const void *v2)
{
  IPAIR *ip1 = (IPAIR *) v1;
  IPAIR *ip2 = (IPAIR *) v2;
  return (ip1->id - ip2->id);
}

int translate_rwforc(int handle)
{
  int i,j,k,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[128],dname[32];
  int *ids,*setid;
  int *nwalls,maxwall;
  float time;
  float *fx,*fy,*fz,*fn;
  float *sfx,*sfy,*sfz,tx,ty,tz;
  int *ftid,*ftns,*ftwall,ncycle,nft;
  IPAIR *ftsort;
  float *ftx,*fty,*ftz;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/rwforc/forces/metadata") == -1) return 0;
  printf("Extracting RWFORC data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num=length;
  ids    = (int *) malloc(num*sizeof(int));
  nwalls = (int *) malloc(num*sizeof(int));
  setid  = (int *) malloc(num*sizeof(int));
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
  lsda_read(handle,LSDA_INT,"setid",0,num,setid);
/*
  Check for force transducers
*/
  if (lsda_cd(handle,"/rwforc/transducer/metadata") == -1) {
    nft=0;
  } else {
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    nft=length;
    ftid   = (int *)   malloc(nft*sizeof(int));
    ftns   = (int *)   malloc(nft*sizeof(int));
    ftwall = (int *)   malloc(nft*sizeof(int));
    ftx    = (float *) malloc(nft*sizeof(int));
    fty    = (float *) malloc(nft*sizeof(int));
    ftz    = (float *) malloc(nft*sizeof(int));
/* and read in metadata for them while we are here */
    lsda_read(handle,LSDA_INTEGER,"ids",0,nft,ftid);
    lsda_read(handle,LSDA_INTEGER,"nodeset",0,nft,ftns);
    lsda_read(handle,LSDA_INTEGER,"rigidwall",0,nft,ftwall);
/*
   resort pointer: to make sure the data is output in increasing order of
   transducer ID.  MPP already does this, but SMP may not, and it is easy
   enough to do here rather than in dyna
*/
    ftsort = (IPAIR *) malloc(nft*sizeof(IPAIR));
    for (i=0; i<nft; i++) {
      ftsort[i].id = ftid[i];
      ftsort[i].pos = i;
    }
    qsort(ftsort,nft,sizeof(IPAIR),ipsort);
  }
/*
  allocate memory to read in 1 state
*/
  fx   = (float *) malloc(num*sizeof(float));
  fy   = (float *) malloc(num*sizeof(float));
  fz   = (float *) malloc(num*sizeof(float));
  fn   = (float *) malloc(num*sizeof(float));
/*
  see which if any walls have segments defined for them...
*/
  for(i=maxwall=0; i<num; i++) {
    sprintf(dirname,"/rwforc/wall%3.3d/metadata/ids",i+1);
    lsda_queryvar(handle,dirname,&typid,&length,&filenum);
    if(typid > 0)
      nwalls[i] = length;
    else
      nwalls[i] = 0;
    if(maxwall < nwalls[i]) maxwall=nwalls[i];
  }
  if(maxwall > 0) {
    sfx   = (float *) malloc(maxwall*sizeof(float));
    sfy   = (float *) malloc(maxwall*sizeof(float));
    sfz   = (float *) malloc(maxwall*sizeof(float));
  }
/*
  open file and write header
*/
  sprintf(output_file,"%srwforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/rwforc/forces/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/rwforc/forces",dp,dname)) != NULL; state++) {
    if(state==1 || nft > 0) {
      fprintf(fp,"\n\n");
      fprintf(fp,"    time       wall#   normal-force    x-force        y-force        z-force\n");
    }
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_force",0,num,fx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_force",0,num,fy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_force",0,num,fz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"normal_force",0,num,fn) != num) break;
    for(i=0; i<num; i++) {
      fprintf(fp,"%12.5E%8d%15.6E%15.6E%15.6E%15.6E\n",
        time,setid[i],fn[i],fx[i],fy[i],fz[i]);
      if(nwalls[i] > 0) {
        sprintf(dirname,"/rwforc/wall%3.3d/%s",i+1,dname);
        lsda_cd(handle,dirname);
        if(lsda_read(handle,LSDA_FLOAT,"x_force",0,nwalls[i],sfx) != nwalls[i]) break;
        if(lsda_read(handle,LSDA_FLOAT,"y_force",0,nwalls[i],sfy) != nwalls[i]) break;
        if(lsda_read(handle,LSDA_FLOAT,"z_force",0,nwalls[i],sfz) != nwalls[i]) break;
        if(lsda_read(handle,LSDA_FLOAT,"total_x",0,1,&tx) != 1) break;
        if(lsda_read(handle,LSDA_FLOAT,"total_y",0,1,&ty) != 1) break;
        if(lsda_read(handle,LSDA_FLOAT,"total_z",0,1,&tz) != 1) break;
        fprintf(fp,"                seg#    \n");
        for(j=0; j<nwalls[i]; j++) {
          fprintf(fp,"            %8d               %15.6E%15.6E%15.6E\n",
            j+1,sfx[j],sfy[j],sfz[j]);
        }
        fprintf(fp,"  total force                      %15.6E%15.6E%15.6E\n",
            tx,ty,tz);
      }
    }
    if(nft > 0) {
      sprintf(dirname,"/rwforc/transducer/%s",dname);
      lsda_cd(handle,dirname);
      if(lsda_read(handle,LSDA_INTEGER,"cycle",0,1,&ncycle) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"x_force",0,nft,ftx) != nft) break;
      if(lsda_read(handle,LSDA_FLOAT,"y_force",0,nft,fty) != nft) break;
      if(lsda_read(handle,LSDA_FLOAT,"z_force",0,nft,ftz) != nft) break;
      fprintf(fp,"\n\n\n");
      fprintf(fp,"r i g i d   w a l l   n o d e   s e t   f o r c e   s u m m a t i o n s\n");
      fprintf(fp,"f o r   t i m e   s t e p %9d   ( at time %12.5E )\n\n",ncycle,time);
      fprintf(fp,"    time       wall #   transducer node set #    x-force      y-force      z-force\n");
      for(j=0; j<nft; j++) {
        k=ftsort[j].pos;
        fprintf(fp,"%12.5E %10d %10d %10d %12.5E %12.5E %12.5E\n",
          time,ftwall[k],ftid[k],ftns[k],ftx[k],fty[k],ftz[k]);
      }
      fprintf(fp,"\n\n");
    }
  }
  fclose(fp);
  if(nft > 0) {
    free(ftsort);
    free(ftz);
    free(fty);
    free(ftx);
    free(ftwall);
    free(ftns);
    free(ftid);
  }
  if(maxwall > 0) {
    free(sfz);
    free(sfy);
    free(sfx);
  }
  free(setid);
  free(fn);
  free(fz);
  free(fy);
  free(fx);
  free(nwalls);
  free(ids);
  printf("      %d states extracted\n",state-1);
}
/*
  NODOUT file
*/
int translate_nodout2(char *base, int handle)
{
  int i,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[128];
  long *ids;
  int cycle,have_rot;
  double time;
  float *x,*y,*z;
  float *x_d,*y_d,*z_d;
  float *x_v,*y_v,*z_v;
  float *x_a,*y_a,*z_a;
  FILE *fp;
  LSDADir *dp = NULL;

  sprintf(dirname,"%s/metadata",base);
  if (lsda_cd(handle,dirname) == -1) return 0;
  printf("Extracting NODOUT data\n");

  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num=length;

/*
  allocate memory to read in 1 state
*/
  ids = (long *) malloc(num*sizeof(long));
  x   = (float *) malloc(num*sizeof(float));
  y   = (float *) malloc(num*sizeof(float));
  z   = (float *) malloc(num*sizeof(float));
  x_d = (float *) malloc(num*sizeof(float));
  y_d = (float *) malloc(num*sizeof(float));
  z_d = (float *) malloc(num*sizeof(float));
  x_v = (float *) malloc(num*sizeof(float));
  y_v = (float *) malloc(num*sizeof(float));
  z_v = (float *) malloc(num*sizeof(float));
  x_a = (float *) malloc(num*sizeof(float));
  y_a = (float *) malloc(num*sizeof(float));
  z_a = (float *) malloc(num*sizeof(float));
  lsda_queryvar(handle,"../d000001/rx_displacement",&typid,&length,&filenum);
  if(typid > 0) {
    have_rot = 1;
  } else {
    have_rot = 0;
  }
/*
  Read metadata
*/
  lsda_read(handle,LSDA_LONG,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%s%s",output_path,base+1);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,dirname,fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one
*/
  for(state=1; next_dir_6or8digitd(handle,base,state) != 0; state++) {
    if(lsda_read(handle,LSDA_DOUBLE,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_displacement",0,num,x_d) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_displacement",0,num,y_d) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_displacement",0,num,z_d) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_velocity",0,num,x_v) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_velocity",0,num,y_v) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_velocity",0,num,z_v) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_acceleration",0,num,x_a) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_acceleration",0,num,y_a) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_acceleration",0,num,z_a) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_coordinate",0,num,x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_coordinate",0,num,y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_coordinate",0,num,z) != num) break;

    fprintf(fp,"\n\n\n n o d a l   p r i n t   o u t   f o r   t i m e  ");
    fprintf(fp,"s t e p%8d                              ( at time%14.7E )\n",cycle,time);
    fprintf(fp,"\n nodal point  x-disp     y-disp      z-disp      ");
    fprintf(fp,"x-vel       y-vel       z-vel      x-accl      y-accl      ");
    fprintf(fp,"z-accl      x-coor      y-coor      z-coor\n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%9ld%13.4E%12.4E%12.4E",ids[i],x_d[i],y_d[i],z_d[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E",x_v[i],y_v[i],z_v[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E",x_a[i],y_a[i],z_a[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",x[i],y[i],z[i]);
    }
    if(have_rot) {
      if(lsda_read(handle,LSDA_FLOAT,"rx_displacement",0,num,x_d) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"ry_displacement",0,num,y_d) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"rz_displacement",0,num,z_d) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"rx_velocity",0,num,x_v) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"ry_velocity",0,num,y_v) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"rz_velocity",0,num,z_v) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"rx_acceleration",0,num,x_a) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"ry_acceleration",0,num,y_a) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"rz_acceleration",0,num,z_a) != num) break;

      fprintf(fp,"\n\n\n n o d a l   p r i n t   o u t   f o r   t i m e  ");
      fprintf(fp,"s t e p%8d                              ( at time%14.7E )\n",cycle,time);
      fprintf(fp,"\n nodal point  x-rot      y-rot       z-rot       ");
      fprintf(fp,"x-rot vel   y-rot vel   z-rot vel   x-rot acc   y-rot acc   ");
      fprintf(fp,"z-rot acc\n");
      for(i=0; i<num; i++) {
        fprintf(fp,"%9ld%13.4E%12.4E%12.4E",ids[i],x_d[i],y_d[i],z_d[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E",x_v[i],y_v[i],z_v[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E\n",x_a[i],y_a[i],z_a[i]);
      }
    }
  }
  fclose(fp);
  free(z_a);
  free(y_a);
  free(x_a);
  free(z_v);
  free(y_v);
  free(x_v);
  free(z_d);
  free(y_d);
  free(x_d);
  free(z);
  free(y);
  free(x);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
int translate_nodout(int handle)
{
  return translate_nodout2("/nodout",handle);
}
int translate_nodouthf(int handle)
{
  return translate_nodout2("/nodouthf",handle);
}
/*
  ELOUTDET file
*/
int translate_eloutdet(int handle)
{
  int i,j,k,typid,filenum,state,intsts,intstn,nodsts,nodstn;
  LSDA_Length length;
  char dirname[256];
  int have_solid, have_tshell, have_nodavg, have_shell;
  FILE *fp;
  MDSOLID solid;
  MDTSHELL tshell;
  MDNODAVG nodavg;
  MDSHELL shell;
  char title_location[128];

  if (lsda_cd(handle,"/eloutdet") == -1) return 0;

  lsda_queryvar(handle,"/eloutdet/solid",&typid,&length,&filenum);
  have_solid= (typid >= 0);
  lsda_queryvar(handle,"/eloutdet/thickshell",&typid,&length,&filenum);
  have_tshell= (typid >= 0);
  lsda_queryvar(handle,"/eloutdet/nodavg",&typid,&length,&filenum);
  have_nodavg= (typid >= 0);
  lsda_queryvar(handle,"/eloutdet/shell",&typid,&length,&filenum);
  have_shell= (typid >= 0);

  title_location[0]=0;
/*
  Read metadata

  Solids
*/
  if(have_solid) {
    lsda_cd(handle,"/eloutdet/solid/metadata");
    strcpy(title_location,"/eloutdet/solid/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        solid.states[j][k]=0;
        j++;
        k=0;
      } else {
        solid.states[j][k++]=dirname[i];
      }
    }
    solid.states[j][k]=0;
    solid.idsize = -1;
    solid.ids = NULL;
    solid.locats=NULL;
    solid.locatn=NULL;
    lsda_read(handle,LSDA_INT,"intsts",0,1,&intsts);
    lsda_read(handle,LSDA_INT,"nodsts",0,1,&nodsts);
    lsda_read(handle,LSDA_INT,"intstn",0,1,&intstn);
    lsda_read(handle,LSDA_INT,"nodstn",0,1,&nodstn);
  }
/*
  thick shells
*/
  if(have_tshell) {
    lsda_cd(handle,"/eloutdet/thickshell/metadata");
    strcpy(title_location,"/eloutdet/thickshell/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        tshell.states[j][k]=0;
        j++;
        k=0;
      } else {
        tshell.states[j][k++]=dirname[i];
      }
    }
    tshell.states[j][k]=0;
    lsda_read(handle,LSDA_I1,"system",0,6,tshell.system);
    tshell.system[6]=0;
    lsda_read(handle,LSDA_INT,"intsts",0,1,&intsts);
    lsda_read(handle,LSDA_INT,"nodsts",0,1,&nodsts);
    lsda_read(handle,LSDA_INT,"intstn",0,1,&intstn);
    lsda_read(handle,LSDA_INT,"nodstn",0,1,&nodstn);
  }
/*
  shells
*/
  if(have_shell) {
    lsda_cd(handle,"/eloutdet/shell/metadata");
    strcpy(title_location,"/eloutdet/shell/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        shell.states[j][k]=0;
        j++;
        k=0;
      } else {
        shell.states[j][k++]=dirname[i];
      }
    }
    shell.states[j][k]=0;
    lsda_read(handle,LSDA_I1,"system",0,6,shell.system);
    shell.system[6]=0;
    for(i=5; i>0 && shell.system[i] == ' '; i--)
      shell.system[i]=0;
    shell.idsize = -1;
    shell.dsize = -1;
    shell.ids = NULL;
    shell.npl = NULL;
    shell.lxx = NULL;
    shell.exx = NULL;
    shell.uxx = NULL;
    shell.sxx = NULL;
    shell.npl = NULL;
    lsda_read(handle,LSDA_INT,"intsts",0,1,&intsts);
    lsda_read(handle,LSDA_INT,"nodsts",0,1,&nodsts);
    lsda_read(handle,LSDA_INT,"intstn",0,1,&intstn);
    lsda_read(handle,LSDA_INT,"nodstn",0,1,&nodstn);
  }
/*
  nodavg
*/
  if(have_nodavg) {
    lsda_cd(handle,"/eloutdet/nodavg/metadata");
    strcpy(title_location,"/eloutdet/nodavg/metadata");
    lsda_read(handle,LSDA_I1,"system",0,6,tshell.system);
  }

  if(strlen(title_location) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  printf("Extracting ELOUTDET data\n");
  sprintf(output_file,"%seloutdet",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  k = 0;
  if(have_solid) {
    lsda_cd(handle,"/eloutdet/solid/metadata");
    i = (have_tshell | have_shell | have_nodavg) ? 0 : 1;
    output_legend(handle,fp,1,i);
    k = 1;
  }
  if(have_tshell) {
    lsda_cd(handle,"/eloutdet/thickshell/metadata");
    i = !k;
    j = (have_shell | have_nodavg) ? 0 : 1;
    output_legend(handle,fp,i,j);
    k = 1;
  }
  if(have_shell) {
    lsda_cd(handle,"/eloutdet/shell/metadata");
    i = !k;
    j = (have_nodavg) ? 0 : 1;
    output_legend(handle,fp,i,j);
    k = 1;
  }
  if(have_nodavg) {
    lsda_cd(handle,"/eloutdet/nodavg/metadata");
    i = !k;
    output_legend(handle,fp,i,1);
  }
/*
  Loop through time states and write each one
*/
  for(state=1;have_solid || have_tshell || have_shell || have_nodavg ; state++) {
    if(have_solid) {
      if(! eloutdet_solid(fp,handle,state,intsts,intstn,nodsts,nodstn,&solid)){
        if(solid.ids) {
          free(solid.ids);
          free(solid.mat);
          free(solid.nip);
          free(solid.nqt);
          if (solid.locats) {
            free(solid.state);
            free(solid.sxx);
            free(solid.syy);
            free(solid.szz);
            free(solid.sxy);
            free(solid.syz);
            free(solid.szx);
            free(solid.yield);
            free(solid.effsg);
            free(solid.locats);
          }
          if (solid.locatn) {
            free(solid.locatn);
          }
        }
        have_solid = 0;
      }
    }
    if(have_tshell) {
      if(! eloutdet_tshell(fp,handle,state,intsts,intstn,nodsts,nodstn,&tshell)) {
        have_tshell = 0;
      }
    }
    if(have_shell) {
      if(! eloutdet_shell(fp,handle,state,intsts,intstn,nodsts,nodstn,&shell)) {
        if(shell.ids) {
          free(shell.ids);
          free(shell.mat);
          free(shell.nip);
          free(shell.npl);
          free(shell.nqt);
          free(shell.iop);
          if(shell.lxx) {
            free(shell.lxx);
            free(shell.lyy);
            free(shell.lzz);
            free(shell.lxy);
            free(shell.lyz);
            free(shell.lzx);
          }
          if(shell.uxx) {
            free(shell.uxx);
            free(shell.uyy);
            free(shell.uzz);
            free(shell.uxy);
            free(shell.uyz);
            free(shell.uzx);
            free(shell.locatn);
          }
          if(shell.sxx) {
            free(shell.sxx);
            free(shell.syy);
            free(shell.szz);
            free(shell.sxy);
            free(shell.syz);
            free(shell.szx);
            free(shell.ps);
            free(shell.state);
            free(shell.locats);
          }
          if(shell.exx) {
            free(shell.exx);
            free(shell.eyy);
            free(shell.ezz);
            free(shell.exy);
            free(shell.eyz);
            free(shell.ezx);
            free(shell.locatn);
          }
        }
        have_shell=0;
      }
    }
    if(have_nodavg) {
      if(! eloutdet_nodavg(fp,handle,state,&nodavg)) {
        if(nodavg.ids) {
          free(nodavg.ids);
          free(nodavg.factor);
          free(nodavg.lxx);
          free(nodavg.lyy);
          free(nodavg.lzz);
          free(nodavg.lxy);
          free(nodavg.lyz);
          free(nodavg.lzx);
          free(nodavg.lyield);
          free(nodavg.uxx);
          free(nodavg.uyy);
          free(nodavg.uzz);
          free(nodavg.uxy);
          free(nodavg.uyz);
          free(nodavg.uzx);
          free(nodavg.uyield);
        }
        have_nodavg = 0;
      }
    }
  }
  fclose(fp);
/*
  free everything here....
*/
  if(have_solid && solid.ids) {
    free(solid.ids);
    free(solid.mat);
    free(solid.nip);
    free(solid.nqt);
    free(solid.state);
    free(solid.sxx);
    free(solid.syy);
    free(solid.szz);
    free(solid.sxy);
    free(solid.syz);
    free(solid.szx);
    free(solid.yield);
    free(solid.effsg);
    free(solid.locatn);
    free(solid.locats);
  }
  if(have_nodavg && nodavg.ids) {
    free(nodavg.ids);
    free(nodavg.factor);
    free(nodavg.lxx);
    free(nodavg.lyy);
    free(nodavg.lzz);
    free(nodavg.lxy);
    free(nodavg.lyz);
    free(nodavg.lzx);
    free(nodavg.lyield);
    free(nodavg.uxx);
    free(nodavg.uyy);
    free(nodavg.uzz);
    free(nodavg.uxy);
    free(nodavg.uyz);
    free(nodavg.uzx);
    free(nodavg.uyield);
  }
  if(have_shell && shell.ids) {
    free(shell.ids);
    free(shell.mat);
    free(shell.nip);
    free(shell.state);
    free(shell.iop);
    if(shell.npl) free(shell.npl);
    if(shell.lxx) {
      free(shell.lxx);
      free(shell.lyy);
      free(shell.lzz);
      free(shell.lxy);
      free(shell.lyz);
      free(shell.lzx);
    }
    if(shell.uxx) {
      free(shell.uxx);
      free(shell.uyy);
      free(shell.uzz);
      free(shell.uxy);
      free(shell.uyz);
      free(shell.uzx);
    }
    free(shell.sxx);
    free(shell.syy);
    free(shell.szz);
    free(shell.sxy);
    free(shell.syz);
    free(shell.szx);
    free(shell.ps);
  }
  printf("      %d states extracted\n",state-1);
  return 0;
}

int
eloutdet_solid(FILE *fp,int handle, int state, int intsts,int intstn,
int nodsts,int nodstn,MDSOLID *solid)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length;
  int num, nums, numn;
  int have_strain,have_stress,i,j,k;

  if(state<=999999)
    sprintf(dirname,"/eloutdet/solid/d%6.6d",state);
  else
    sprintf(dirname,"/eloutdet/solid/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"sig_xx",&typid,&length,&filenum);
  have_stress = (typid > 0);
  lsda_queryvar(handle,"eps_xx",&typid,&length,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
  num=length;
/* element information */
  solid->ids = (int *) malloc(num*sizeof(int));
  solid->mat = (int *) malloc(num*sizeof(int));
  solid->nip = (int *) malloc(num*sizeof(int));
  solid->nqt = (int *) malloc(num*sizeof(int));
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,solid->ids) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"mat",0,length,solid->mat) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"nip",0,length,solid->nip) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"nqt",0,length,solid->nqt) != num) return 0;
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
  fprintf(fp," element  materl\n");
/* stress */
  nums=0;
  if (have_stress) {
    lsda_queryvar(handle,"locats",&typid,&length,&filenum);
    nums=length;
    solid->state = (int *) malloc(nums*sizeof(int));
    solid->sxx = (float *) malloc(nums*sizeof(float));
    solid->syy = (float *) malloc(nums*sizeof(float));
    solid->szz = (float *) malloc(nums*sizeof(float));
    solid->sxy = (float *) malloc(nums*sizeof(float));
    solid->syz = (float *) malloc(nums*sizeof(float));
    solid->szx = (float *) malloc(nums*sizeof(float));
    solid->yield = (float *) malloc(nums*sizeof(float));
    solid->effsg = (float *) malloc(nums*sizeof(float));
    solid->locats = (int *) malloc(nums*sizeof(int));
    if(lsda_read(handle,LSDA_INT,"state",0,nums,solid->state) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xx",0,nums,solid->sxx) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yy",0,nums,solid->syy) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zz",0,nums,solid->szz) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xy",0,nums,solid->sxy) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yz",0,nums,solid->syz) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zx",0,nums,solid->szx) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"yield",0,nums,solid->yield) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"effsg",0,nums,solid->effsg) != nums) return 0;
    if(lsda_read(handle,LSDA_INT,"locats",0,nums,solid->locats) != nums) return 0;
    fprintf(fp,"     ipt  stress       sig-xx      sig-yy      sig");
    fprintf(fp,"-zz      sig-xy      sig-yz      sig-zx           ");
    fprintf(fp,"            yield      location\n           state                    ");
    fprintf(fp,"                                                              ");
    fprintf(fp,"effsg      function\n");
    for(i=k=0; i<num; i++) {
      fprintf(fp,"%8d-%7d\n",solid->ids[i],solid->mat[i]);
      if (intsts==1) {
        for(j=0; j<solid->nip[i]; j++,k++) {
          fprintf(fp,"         %-7s ",solid->states[solid->state[k]-1]);
          fprintf(fp,"%12.4E%12.4E%12.4E",solid->sxx[k],solid->syy[k],solid->szz[k]);
          fprintf(fp,"%12.4E%12.4E%12.4E%14.4E%14.4E      int. point%3d\n",
          solid->sxy[k],solid->syz[k],solid->szx[k],solid->effsg[k],
          solid->yield[k],solid->locats[k]);
        }
      }
      if (nodsts==1) {
        for(j=0; j<solid->nqt[i]; j++,k++) {
          fprintf(fp,"         %-7s ",solid->states[solid->state[k]-1]);
          fprintf(fp,"%12.4E%12.4E%12.4E",solid->sxx[k],solid->syy[k],solid->szz[k]);
          fprintf(fp,"%12.4E%12.4E%12.4E%14.4E%14.4E      node%9d\n",
          solid->sxy[k],solid->syz[k],solid->szx[k],solid->effsg[k],
          solid->yield[k],solid->locats[k]);
        }
      }
    }
  }
  numn=0;
  if (have_strain) {
    lsda_queryvar(handle,"locatn",&typid,&length,&filenum);
    numn=length;
    solid->locatn = (int *) malloc(numn*sizeof(int));
    if (numn>nums) {
      solid->sxx = (float *) malloc(numn*sizeof(float));
      solid->syy = (float *) malloc(numn*sizeof(float));
      solid->szz = (float *) malloc(numn*sizeof(float));
      solid->sxy = (float *) malloc(numn*sizeof(float));
      solid->syz = (float *) malloc(numn*sizeof(float));
      solid->szx = (float *) malloc(numn*sizeof(float));
    }
    if(lsda_read(handle,LSDA_FLOAT,"eps_xx",0,numn,solid->sxx) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yy",0,numn,solid->syy) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zz",0,numn,solid->szz) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_xy",0,numn,solid->sxy) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yz",0,numn,solid->syz) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zx",0,numn,solid->szx) != numn) return 0;
    if(lsda_read(handle,LSDA_INT,"locatn",0,numn,solid->locatn) != numn) return 0;
    fprintf(fp,"\n\n\n e l e m e n t   s t r a i n   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
    fprintf(fp," element  strain       eps-xx      eps-yy      eps");
    fprintf(fp,"-zz      eps-xy      eps-yz      eps-zx      location\n");
    fprintf(fp," num/ipt   state\n");
    for(i=k=0; i<num; i++) {
      fprintf(fp,"%8d-%7d\n",solid->ids[i],solid->mat[i]);
      if (intstn==1) {
        for(j=0; j<solid->nip[i]; j++,k++) {
          fprintf(fp,"                 ");
          fprintf(fp,"%12.4E%12.4E%12.4E",solid->sxx[k],solid->syy[k],solid->szz[k]);
          fprintf(fp,"%12.4E%12.4E%12.4E      int. point%3d\n",solid->sxy[k],
                  solid->syz[k],solid->szx[k],solid->locatn[k]);
        }
      }
      if (nodstn==1) {
        for(j=0; j<solid->nqt[i]; j++,k++) {
          fprintf(fp,"                 ");
          fprintf(fp,"%12.4E%12.4E%12.4E",solid->sxx[k],solid->syy[k],solid->szz[k]);
          fprintf(fp,"%12.4E%12.4E%12.4E      node%9d\n",solid->sxy[k],
                  solid->syz[k],solid->szx[k],solid->locatn[k]);
        }
      }
    }
  }
  return 1;
}

int
eloutdet_tshell(FILE *fp,int handle, int state,  int intsts,int intstn,
int nodsts,int nodstn,MDTSHELL *tshell)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length;
  int num,numn,nums;
  int have_strain,have_stress,i,j,k,i1;

  if(state<=999999)
    sprintf(dirname,"/eloutdet/thickshell/d%6.6d",state);
  else
    sprintf(dirname,"/eloutdet/thickshell/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"sig_xx",&typid,&length,&filenum);
  have_stress = (typid > 0);
  lsda_queryvar(handle,"eps_xx",&typid,&length,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
/* read in element information */
  num=length;
  tshell->ids = (int *) malloc(num*sizeof(int));
  tshell->mat = (int *) malloc(num*sizeof(int));
  tshell->nip = (int *) malloc(num*sizeof(int));
  tshell->npl = (int *) malloc(num*sizeof(int));
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,num,tshell->ids) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"mat",0,num,tshell->mat) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"nip",0,num,tshell->nip) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"npl",0,num,tshell->npl) != num) return 0;
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
  fprintf(fp," element  materl\n");
/* read in stress */
  if (have_stress) {
    lsda_queryvar(handle,"locats",&typid,&length,&filenum);
    if(typid < 0) return 0;  /* all elements deleted */
    nums=length; 
    tshell->state = (int *) malloc(nums*sizeof(int));
    tshell->sxx = (float *) malloc(nums*sizeof(float));
    tshell->syy = (float *) malloc(nums*sizeof(float));
    tshell->szz = (float *) malloc(nums*sizeof(float));
    tshell->sxy = (float *) malloc(nums*sizeof(float));
    tshell->syz = (float *) malloc(nums*sizeof(float));
    tshell->szx = (float *) malloc(nums*sizeof(float));
    tshell->yield = (float *) malloc(nums*sizeof(float));
    tshell->effsg = (float *) malloc(nums*sizeof(float));
    tshell->locats = (int *) malloc(nums*sizeof(int));
    if(lsda_read(handle,LSDA_INT,"state",0,nums,tshell->state) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xx",0,nums,tshell->sxx) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yy",0,nums,tshell->syy) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zz",0,nums,tshell->szz) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xy",0,nums,tshell->sxy) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yz",0,nums,tshell->syz) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zx",0,nums,tshell->szx) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"yield",0,nums,tshell->yield) != nums) return 0;
    if(lsda_read(handle,LSDA_INT,"locats",0,nums,tshell->locats) != nums) return 0;
/*
    Output stress    
*/  
    fprintf(fp," num/ipt  stress       sig-xx      sig-yy      sig");
    fprintf(fp,"-zz      sig-xy      sig-yz      sig-zx         ");
    fprintf(fp,"yield      location\n           state                    ");
    fprintf(fp,"                                                ");
    fprintf(fp,"           function\n");
    for(j=k=0;j<num;j++){
      fprintf(fp,"%8d-%5d\n",tshell->ids[j],tshell->mat[j]);
      if (intsts==1) {
        for(i=0; i<tshell->nip[j]; i++) {
          for(i1=0; i1<tshell->npl[j]; i1++,k++) {
            fprintf(fp,"      %d  %-7s ",i+1,tshell->states[tshell->state[k]-1]);
            fprintf(fp,"%12.4E%12.4E%12.4E",tshell->sxx[k],tshell->syy[k],
                    tshell->szz[k]);
            fprintf(fp,"%12.4E%12.4E%12.4E%14.4E  int. point%3d\n",tshell->sxy[k],
                    tshell->syz[k],tshell->szx[k],tshell->yield[k],i1+1);
          }
        }
      }
      if (nodsts==1) {
        for(i=0; i<8; i++,k++) {
          fprintf(fp,"         %-7s ",tshell->states[tshell->state[k]-1]);
          fprintf(fp,"%12.4E%12.4E%12.4E",tshell->sxx[k],tshell->syy[k],
                  tshell->szz[k]);
          fprintf(fp,"%12.4E%12.4E%12.4E%14.4E  node %8d\n",tshell->sxy[k],
                  tshell->syz[k],tshell->szx[k],tshell->yield[k],tshell->locats[k]);
        }
      }
    }
  }
  if (have_strain) {
    lsda_queryvar(handle,"locatn",&typid,&length,&filenum);
    if(typid < 0) return 0;  /* all elements deleted */
    numn=length; 
    tshell->exx = (float *) malloc(numn*sizeof(float));
    tshell->eyy = (float *) malloc(numn*sizeof(float));
    tshell->ezz = (float *) malloc(numn*sizeof(float));
    tshell->exy = (float *) malloc(numn*sizeof(float));
    tshell->eyz = (float *) malloc(numn*sizeof(float));
    tshell->ezx = (float *) malloc(numn*sizeof(float));
    tshell->locatn = (int *) malloc(numn*sizeof(int)); 
    if(lsda_read(handle,LSDA_FLOAT,"eps_xx",0,numn,tshell->exx) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yy",0,numn,tshell->eyy) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zz",0,numn,tshell->ezz) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_xy",0,numn,tshell->exy) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yz",0,numn,tshell->eyz) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zx",0,numn,tshell->ezx) != numn) return 0;
    if(lsda_read(handle,LSDA_INT,"locatn",0,numn,tshell->locatn) != numn) return 0;
    fprintf(fp,"\n num/ipt  strain");
    fprintf(fp,"      eps-xx      eps-yy      eps");
    fprintf(fp,"-zz      eps-xy      eps-yz      eps-zx      location                      \n\n ");
    for(j=k=0;j<num;j++){
      fprintf(fp,"%8d-%5d\n",tshell->ids[j],tshell->mat[j]);
      if (intstn==1) {
        for(i=0; i<tshell->nip[j]; i++) {
          for(i1=0; i1<tshell->npl[j]; i1++,k++) {
            fprintf(fp,"%4d-            %12.4E%12.4E%12.4E",i+1,
                    tshell->exx[k],tshell->eyy[k],tshell->ezz[k]);
            fprintf(fp,"%12.4E%12.4E%12.4E   int. point%3d\n",tshell->exy[k],
                  tshell->eyz[k],tshell->ezx[k],i1+1);
          }
        }
      }
      if (nodstn==1) {
        for(i=0; i<8; i++,k++) {
          fprintf(fp,"                 %12.4E%12.4E%12.4E",
                  tshell->exx[k],tshell->eyy[k],tshell->ezz[k]);
          fprintf(fp,"%12.4E%12.4E%12.4E   node %8d\n",tshell->exy[k],
                  tshell->eyz[k],tshell->ezx[k],tshell->locatn[k]);
        }
      }
    }
  }
  free(tshell->ids);
  free(tshell->mat);
  free(tshell->nip);
  free(tshell->npl);
  if (have_stress) {
     free(tshell->state);
     free(tshell->sxx);
     free(tshell->syy);
     free(tshell->szz);
     free(tshell->sxy);
     free(tshell->syz);
     free(tshell->szx);
     free(tshell->yield);
     free(tshell->effsg);
     free(tshell->locats);
  }
  if (have_strain) {
     free(tshell->exx);
     free(tshell->eyy);
     free(tshell->ezz);
     free(tshell->exy);
     free(tshell->eyz);
     free(tshell->ezx);
     free(tshell->locatn);
  }
  return 1;
}

int
eloutdet_shell(FILE *fp,int handle, int state, int intsts, int intstn,
int nodsts,int nodstn,MDSHELL *shell)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length,length2;
  int num,numn,nums;
  int have_strain,have_stress,i,j,k,j1;

  if(state<=999999)
    sprintf(dirname,"/eloutdet/shell/d%6.6d",state);
  else
    sprintf(dirname,"/eloutdet/shell/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"sig_xx",&typid,&length2,&filenum);
  have_stress = (typid > 0);
  lsda_queryvar(handle,"eps_xx",&typid,&length2,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;
  num=length;
  if (have_stress) {
    lsda_queryvar(handle,"locats",&typid,&length,&filenum);
    nums=length;
    } else {
    nums=0;
  }
  if (have_strain) {
    lsda_queryvar(handle,"locatn",&typid,&length,&filenum);
     numn=length;
    } else {
    numn=0;
  }

  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  shell->ids = (int *) malloc(num*sizeof(int));
  shell->mat = (int *) malloc(num*sizeof(int));
  shell->nip = (int *) malloc(num*sizeof(int));
  shell->iop = (int *) malloc(num*sizeof(int));
  shell->npl = (int *) malloc(num*sizeof(int));
  shell->nqt = (int *) malloc(num*sizeof(int));
  if(lsda_read(handle,LSDA_INT,"ids",0,num,shell->ids) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"mat",0,num,shell->mat) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"nip",0,num,shell->nip) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"iop",0,num,shell->iop) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"npl",0,num,shell->npl) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"nqt",0,num,shell->nqt) != num) return 0;
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
  sprintf(dirname,"(%s)",shell->system);
  fprintf(fp," element  materl%-8s\n",dirname);
  if(have_stress) {
    shell->state = (int *) malloc(nums*sizeof(int));
    shell->sxx = (float *) malloc(nums*sizeof(float));
    shell->syy = (float *) malloc(nums*sizeof(float));
    shell->szz = (float *) malloc(nums*sizeof(float));
    shell->sxy = (float *) malloc(nums*sizeof(float));
    shell->syz = (float *) malloc(nums*sizeof(float));
    shell->szx = (float *) malloc(nums*sizeof(float));
    shell->ps  = (float *) malloc(nums*sizeof(float));
    shell->locats = (int *) malloc(nums*sizeof(int));
    if(lsda_read(handle,LSDA_INT,"state",0,nums,shell->state) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xx",0,nums,shell->sxx) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yy",0,nums,shell->syy) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zz",0,nums,shell->szz) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xy",0,nums,shell->sxy) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yz",0,nums,shell->syz) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zx",0,nums,shell->szx) != nums) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"plastic_strain",0,nums,shell->ps) != nums) return 0;
    if(lsda_read(handle,LSDA_INT,"locats",0,nums,shell->locats) != nums) return 0;
/*
    Output this data
*/
    fprintf(fp," ipt-shl  stress       sig-xx      sig-yy      sig");
    fprintf(fp,"-zz      sig-xy      sig-yz      sig-zx       plastic      location\n");
    fprintf(fp,"           state                                  ");
    fprintf(fp,"                                               strain \n");

    for(i=k=0; i<num; i++) {
      fprintf(fp,"%8d-%7d\n",shell->ids[i],shell->mat[i]);
      for(j=0; j<shell->nip[i]; j++) {
        if (intsts==1) {
          for (j1=0; j1<shell->npl[i]; j1++,k++) {
            fprintf(fp,"%4d-%3d %-7s ",j+1,shell->iop[i],shell->states[shell->state[k]-1]);
            fprintf(fp,"%12.4E%12.4E%12.4E",shell->sxx[k],shell->syy[k],shell->szz[k]);
            fprintf(fp,"%12.4E%12.4E%12.4E%14.4E  int. point%3d\n",shell->sxy[k],
                    shell->syz[k],shell->szx[k],shell->ps[k],j1+1);
          }
        }
        if (nodsts==1) {
          for (j1=0; j1<shell->nqt[i]; j1++,k++) {
            fprintf(fp,"%4d-%3d %-7s ",j+1,shell->iop[i],shell->states[shell->state[k]-1]);
            fprintf(fp,"%12.4E%12.4E%12.4E",shell->sxx[k],shell->syy[k],shell->szz[k]);
            fprintf(fp,"%12.4E%12.4E%12.4E%14.4E  node %8d\n",shell->sxy[k],
                    shell->syz[k],shell->szx[k],shell->ps[k],shell->locats[k]);
          }
        }
      }
    }
  }
  if(have_strain) {
    shell->exx = (float *) malloc(numn*sizeof(float));
    shell->eyy = (float *) malloc(numn*sizeof(float));
    shell->ezz = (float *) malloc(numn*sizeof(float));
    shell->exy = (float *) malloc(numn*sizeof(float));
    shell->eyz = (float *) malloc(numn*sizeof(float));
    shell->ezx = (float *) malloc(numn*sizeof(float));
    shell->locatn = (int *) malloc(numn*sizeof(int));
    if(lsda_read(handle,LSDA_FLOAT,"eps_xx",0,numn,shell->exx) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yy",0,numn,shell->eyy) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zz",0,numn,shell->ezz) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_xy",0,numn,shell->exy) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yz",0,numn,shell->eyz) != numn) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zx",0,numn,shell->ezx) != numn) return 0;
    if(lsda_read(handle,LSDA_INT,"locatn",0,numn,shell->locatn) != numn) return 0;
/*
     all the blanks here are silly, but are output by DYNA so.....
*/
/*  sprintf(dirname,"(%s)",shell->system); */
    fprintf(fp,"\n ipt-shl  strain       eps-xx      eps-yy      eps");
    fprintf(fp,"-zz      eps-xy      eps-yz      eps-zx        location          \n");
    fprintf(fp,"                                                  ");
    fprintf(fp,"                                                                   \n");

    for(i=k=0; i<num; i++) {
      fprintf(fp,"%8d-%7d\n",shell->ids[i],shell->mat[i]);
      for(j=0; j<shell->nip[i]; j++) {
        if (intstn==1) {
          for (j1=0; j1<shell->npl[i]; j1++,k++) {
            fprintf(fp,"%4d-            %12.4E%12.4E%12.4E",j+1,
                    shell->exx[k],shell->eyy[k],shell->ezz[k]);
            fprintf(fp,"%12.4E%12.4E%12.4E   int. point%3d\n",shell->exy[k],
                    shell->eyz[k],shell->ezx[k],j1+1);
          }
        }
        if (nodstn==1) {
          for (j1=0; j1<shell->nqt[i]; j1++,k++) {
            fprintf(fp,"%4d-            %12.4E%12.4E%12.4E",j+1,
                    shell->exx[k],shell->eyy[k],shell->ezz[k]);
            fprintf(fp,"%12.4E%12.4E%12.4E   node %8d\n",shell->exy[k],
                  shell->eyz[k],shell->ezx[k],shell->locatn[k]);
          }
        }
      }
    }
  }
  return 1;
}
int
eloutdet_nodavg(FILE *fp,int handle, int state, MDNODAVG *nodavg) 
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length,length2;
  int num;
  int have_strain,have_stress,i;

  if(state<=999999)
    sprintf(dirname,"/eloutdet/nodavg/d%6.6d",state);
  else
    sprintf(dirname,"/eloutdet/nodavg/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"lower_sig_xx",&typid,&length2,&filenum);
  have_stress = (typid > 0);
  lsda_queryvar(handle,"lower_eps_xx",&typid,&length2,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;
  num=length;

  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  nodavg->ids    = (int *) malloc(num*sizeof(int));
  nodavg->factor = (float *) malloc(num*sizeof(float));
  nodavg->lxx    = (float *) malloc(num*sizeof(float));
  nodavg->lyy    = (float *) malloc(num*sizeof(float));
  nodavg->lzz    = (float *) malloc(num*sizeof(float));
  nodavg->lxy    = (float *) malloc(num*sizeof(float));
  nodavg->lyz    = (float *) malloc(num*sizeof(float));
  nodavg->lzx    = (float *) malloc(num*sizeof(float));
  nodavg->lyield = (float *) malloc(num*sizeof(float));
  nodavg->uxx    = (float *) malloc(num*sizeof(float));
  nodavg->uyy    = (float *) malloc(num*sizeof(float));
  nodavg->uzz    = (float *) malloc(num*sizeof(float));
  nodavg->uxy    = (float *) malloc(num*sizeof(float));
  nodavg->uyz    = (float *) malloc(num*sizeof(float));
  nodavg->uzx    = (float *) malloc(num*sizeof(float));
  nodavg->uyield = (float *) malloc(num*sizeof(float));
  
  if(lsda_read(handle,LSDA_INT,"ids",0,num,nodavg->ids) != num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"factor",0,num,nodavg->factor) != num) return 0;
  if(have_stress) {
    fprintf(fp,"\n\n\n n o d a l  s t r e s s   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time%12.5E )\n\n",cycle,time);
    fprintf(fp," node (global)\n");
    sprintf(dirname,"(%s)",nodavg->system);
    if(lsda_read(handle,LSDA_FLOAT,"lower_sig_xx",0,num,nodavg->lxx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_sig_yy",0,num,nodavg->lyy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_sig_zz",0,num,nodavg->lzz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_sig_xy",0,num,nodavg->lxy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_sig_yz",0,num,nodavg->lyz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_sig_zx",0,num,nodavg->lzx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_yield",0,num,nodavg->lyield) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_sig_xx",0,num,nodavg->uxx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_sig_yy",0,num,nodavg->uyy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_sig_zz",0,num,nodavg->uzz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_sig_xy",0,num,nodavg->uxy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_sig_yz",0,num,nodavg->uyz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_sig_zx",0,num,nodavg->uzx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_sig_zx",0,num,nodavg->uzx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_yield",0,num,nodavg->uyield) != num) return 0;
/*
    Output this data
*/
    fprintf(fp,"          stress       sig-xx      sig-yy      sig");
    fprintf(fp,"-zz      sig-xy      sig-yz      sig-zx       plastic\n");
    fprintf(fp,"                                                  ");
    fprintf(fp,"                                               strain \n");

    for(i=0; i<num; i++) {
      fprintf(fp,"%8d-\n",nodavg->ids[i]);
      if (nodavg->factor[i]>0) {
        fprintf(fp," lower surface   %12.4E%12.4E",nodavg->lxx[i],nodavg->lyy[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E",nodavg->lzz[i],nodavg->lxy[i],nodavg->lyz[i]);
        fprintf(fp,"%12.4E%12.4E\n",nodavg->lzx[i],nodavg->lyield[i]);
        fprintf(fp," upper surface   %12.4E%12.4E",nodavg->uxx[i],nodavg->uyy[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E",nodavg->uzz[i],nodavg->uxy[i],nodavg->uyz[i]);
        fprintf(fp,"%12.4E%12.4E\n",nodavg->uzx[i],nodavg->uyield[i]);}
      else {
        fprintf(fp," mid.  surface   %12.4E%12.4E",nodavg->lxx[i],nodavg->lyy[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E",nodavg->lzz[i],nodavg->lxy[i],nodavg->lyz[i]);
        fprintf(fp,"%12.4E%12.4E\n",nodavg->lzx[i],nodavg->lyield[i]);
      }
    }
  }
  if(have_strain) {
    fprintf(fp,"\n\n\n n o d a l  s t r a i n   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time%12.5E )\n\n",cycle,time);
    fprintf(fp," node (global)\n");
    sprintf(dirname,"(%s)",nodavg->system);
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_xx",0,num,nodavg->lxx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_yy",0,num,nodavg->lyy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_zz",0,num,nodavg->lzz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_xy",0,num,nodavg->lxy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_yz",0,num,nodavg->lyz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_zx",0,num,nodavg->lzx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_xx",0,num,nodavg->uxx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_yy",0,num,nodavg->uyy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_zz",0,num,nodavg->uzz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_xy",0,num,nodavg->uxy) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_yz",0,num,nodavg->uyz) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_zx",0,num,nodavg->uzx) != num) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_zx",0,num,nodavg->uzx) != num) return 0;
/*
    Output this data
*/
    fprintf(fp,"          strain       eps-xx      eps-yy      eps");
    fprintf(fp,"-zz      eps-xy      eps-yz      eps-zx       \n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%8d-\n",nodavg->ids[i]);
      if (nodavg->factor[i]>0) {
        fprintf(fp," lower surface   %12.4E%12.4E",nodavg->lxx[i],nodavg->lyy[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E",nodavg->lzz[i],nodavg->lxy[i],nodavg->lyz[i]);
        fprintf(fp,"%12.4E\n",nodavg->lzx[i]);
        fprintf(fp," upper surface   %12.4E%12.4E",nodavg->uxx[i],nodavg->uyy[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E",nodavg->uzz[i],nodavg->uxy[i],nodavg->uyz[i]);
        fprintf(fp,"%12.4E\n",nodavg->uzx[i]);}
      else {
        fprintf(fp," mid.  surface   %12.4E%12.4E",nodavg->lxx[i],nodavg->lyy[i]);
        fprintf(fp,"%12.4E%12.4E%12.4E",nodavg->lzz[i],nodavg->lxy[i],nodavg->lyz[i]);
        fprintf(fp,"%12.4E\n",nodavg->lzx[i]);
      }
    }
  }
  return 1;
}
/*
  ELOUT file
*/


int translate_elout(int handle)
{
  int i,j,k,typid,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int have_solid, have_tshell, have_beam, have_shell;
  int have_solid_hist, have_tshell_hist, have_beam_hist, have_shell_hist;
  FILE *fp;
  MDSOLID solid;
  MDTSHELL tshell;
  MDBEAM beam;
  MDSHELL shell;
  MDHIST solid_hist,tshell_hist,beam_hist,shell_hist;
  char title_location[128];

  if (lsda_cd(handle,"/elout") == -1) return 0;

  lsda_queryvar(handle,"/elout/solid",&typid,&length,&filenum);
  have_solid= (typid >= 0);
  lsda_queryvar(handle,"/elout/solid_hist",&typid,&length,&filenum);
  have_solid_hist= (typid >= 0);
  lsda_queryvar(handle,"/elout/thickshell",&typid,&length,&filenum);
  have_tshell= (typid >= 0);
  lsda_queryvar(handle,"/elout/thickshell_hist",&typid,&length,&filenum);
  have_tshell_hist= (typid >= 0);
  lsda_queryvar(handle,"/elout/beam",&typid,&length,&filenum);
  have_beam= (typid >= 0);
  lsda_queryvar(handle,"/elout/beam_hist",&typid,&length,&filenum);
  have_beam_hist= (typid >= 0);
  lsda_queryvar(handle,"/elout/shell",&typid,&length,&filenum);
  have_shell= (typid >= 0);
  lsda_queryvar(handle,"/elout/shell_hist",&typid,&length,&filenum);
  have_shell_hist= (typid >= 0);

  title_location[0]=0;
/*
  Read metadata

  Solids
*/
  if(have_solid) {
    lsda_cd(handle,"/elout/solid/metadata");
    strcpy(title_location,"/elout/solid/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        solid.states[j][k]=0;
        j++;
        k=0;
      } else {
        solid.states[j][k++]=dirname[i];
      }
    }
    solid.states[j][k]=0;
    solid.idsize = -1;
    solid.ids = NULL;
  }
/*
  thick shells
*/
  if(have_tshell) {
    lsda_cd(handle,"/elout/thickshell/metadata");
    strcpy(title_location,"/elout/thickshell/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        tshell.states[j][k]=0;
        j++;
        k=0;
      } else {
        tshell.states[j][k++]=dirname[i];
      }
    }
    tshell.states[j][k]=0;
    lsda_read(handle,LSDA_I1,"system",0,6,tshell.system);
    tshell.system[6]=0;
  }
/*
  beams
*/
  if(have_beam) {
    lsda_cd(handle,"/elout/beam/metadata");
    strcpy(title_location,"/elout/beam/metadata");
    beam.idsize = -1;
    beam.dsize = -1;
    beam.ids = NULL;
    beam.s11 = NULL;
  }
/*
  shells
*/
  if(have_shell) {
    lsda_cd(handle,"/elout/shell/metadata");
    strcpy(title_location,"/elout/shell/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        shell.states[j][k]=0;
        j++;
        k=0;
      } else {
        shell.states[j][k++]=dirname[i];
      }
    }
    shell.states[j][k]=0;
    lsda_read(handle,LSDA_I1,"system",0,6,shell.system);
    shell.system[6]=0;
    for(i=5; i>0 && shell.system[i] == ' '; i--)
      shell.system[i]=0;
    shell.idsize = -1;
    shell.dsize = -1;
    shell.ids = NULL;
    shell.npl = NULL;
    shell.lxx = NULL;
    shell.uxx = NULL;
    shell.sxx = NULL;
  }
/*
  HIST types
*/
  if(have_solid_hist) {
    lsda_cd(handle,"/elout/solid_hist/metadata");
    strcpy(title_location,"/elout/solid_hist/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        solid_hist.states[j][k]=0;
        j++;
        k=0;
      } else {
        solid_hist.states[j][k++]=dirname[i];
      }
    }
    solid_hist.states[j][k]=0;
    solid_hist.idsize = -1;
    solid_hist.nhv = -1;
    solid_hist.ids = NULL;
    solid_hist.mat = NULL;
    solid_hist.ndata = NULL;
    solid_hist.nhist = NULL;
    solid_hist.data = NULL;
    solid_hist.hist = NULL;
    solid_hist.strain = NULL;
  }
  if(have_tshell_hist) {
    lsda_cd(handle,"/elout/thickshell_hist/metadata");
    strcpy(title_location,"/elout/thickshell_hist/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        tshell_hist.states[j][k]=0;
        j++;
        k=0;
      } else {
        tshell_hist.states[j][k++]=dirname[i];
      }
    }
    tshell_hist.states[j][k]=0;
    tshell_hist.idsize = -1;
    tshell_hist.nhv = -1;
    tshell_hist.ids = NULL;
    tshell_hist.mat = NULL;
    tshell_hist.ndata = NULL;
    tshell_hist.nhist = NULL;
    tshell_hist.data = NULL;
    tshell_hist.hist = NULL;
    tshell_hist.strain = NULL;
  }
  if(have_beam_hist) {
    lsda_cd(handle,"/elout/beam_hist/metadata");
    strcpy(title_location,"/elout/beam_hist/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        beam_hist.states[j][k]=0;
        j++;
        k=0;
      } else {
        beam_hist.states[j][k++]=dirname[i];
      }
    }
    beam_hist.states[j][k]=0;
    beam_hist.idsize = -1;
    beam_hist.nhv = -1;
    beam_hist.ids = NULL;
    beam_hist.mat = NULL;
    beam_hist.ndata = NULL;
    beam_hist.nhist = NULL;
    beam_hist.data = NULL;
    beam_hist.hist = NULL;
    beam_hist.strain = NULL;
  }
  if(have_shell_hist) {
    lsda_cd(handle,"/elout/shell_hist/metadata");
    strcpy(title_location,"/elout/shell_hist/metadata");
    lsda_queryvar(handle,"states",&typid,&length,&filenum);
    lsda_read(handle,LSDA_I1,"states",0,length,dirname);
    for(i=j=k=0; i<length; i++) {
      if(dirname[i] == ',') {
        shell_hist.states[j][k]=0;
        j++;
        k=0;
      } else {
        shell_hist.states[j][k++]=dirname[i];
      }
    }
    shell_hist.states[j][k]=0;
    shell_hist.idsize = -1;
    shell_hist.nhv = -1;
    shell_hist.ids = NULL;
    shell_hist.mat = NULL;
    shell_hist.ndata = NULL;
    shell_hist.nhist = NULL;
    shell_hist.data = NULL;
    shell_hist.hist = NULL;
    shell_hist.strain = NULL;
  }
  if(strlen(title_location) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  printf("Extracting ELOUT data\n");
  sprintf(output_file,"%selout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  k = 0;
  if(have_solid) {
    lsda_cd(handle,"/elout/solid/metadata");
    i = (have_tshell      | have_beam      | have_shell |
         have_tshell_hist | have_beam_hist | have_shell_hist) ? 0 : 1;
    output_legend(handle,fp,1,i);
    k = 1;
  } else if(have_solid_hist) {
    lsda_cd(handle,"/elout/solid_hist/metadata");
    i = (have_tshell      | have_beam      | have_shell |
         have_tshell_hist | have_beam_hist | have_shell_hist) ? 0 : 1;
    output_legend(handle,fp,1,i);
    k = 1;
  }
  if(have_tshell) {
    lsda_cd(handle,"/elout/thickshell/metadata");
    i = !k;
    j = (have_beam      | have_shell |
         have_beam_hist | have_shell_hist) ? 0 : 1;
    output_legend(handle,fp,i,j);
    k = 1;
  } else if(have_tshell_hist) {
    lsda_cd(handle,"/elout/thickshell_hist/metadata");
    i = !k;
    j = (have_beam      | have_shell |
         have_beam_hist | have_shell_hist) ? 0 : 1;
    output_legend(handle,fp,i,j);
    k = 1;
  }
  if(have_beam) {
    lsda_cd(handle,"/elout/beam/metadata");
    i = !k;
    j = (have_shell | have_shell_hist) ? 0 : 1;
    output_legend(handle,fp,i,j);
    k = 1;
  } else if(have_beam_hist) {
    lsda_cd(handle,"/elout/beam_hist/metadata");
    i = !k;
    j = (have_shell | have_shell_hist) ? 0 : 1;
    output_legend(handle,fp,i,j);
    k = 1;
  }
  if(have_shell) {
    lsda_cd(handle,"/elout/shell/metadata");
    i = !k;
    output_legend(handle,fp,i,1);
  } else if(have_shell_hist) {
    lsda_cd(handle,"/elout/shell_hist/metadata");
    i = !k;
    output_legend(handle,fp,i,1);
  }
/*
  Loop through time states and write each one
*/
  for(state=1;  have_solid  || have_solid_hist  ||
                have_tshell || have_tshell_hist ||
                have_beam   || have_beam_hist   ||
                have_shell  || have_shell_hist    ; state++) {
    if(have_solid) {
      if(! elout_solid(fp,handle,state,&solid)) {
        if(solid.ids) {
          free(solid.ids);
          free(solid.mat);
          free(solid.state);
          free(solid.sxx);
          free(solid.syy);
          free(solid.szz);
          free(solid.sxy);
          free(solid.syz);
          free(solid.szx);
          free(solid.yield);
          free(solid.effsg);
        }
        have_solid = 0;
      }
    } else if(have_solid_hist) {
      if(! elout_solid_hist(fp,handle,state,&solid_hist)) {
        if(solid_hist.ids) {
          free(solid_hist.ids);
          free(solid_hist.mat);
          free(solid_hist.ndata);
          free(solid_hist.nhist);
          free(solid_hist.data);
          free(solid_hist.hist);
          free(solid_hist.strain);
        }
        have_solid_hist = 0;
      }
    }
    if(have_tshell) {
      if(! elout_tshell(fp,handle,state,&tshell)) {
        have_tshell = 0;
      }
    } else if(have_tshell_hist) {
      if(! elout_tshell_hist(fp,handle,state,&tshell_hist)) {
        if(tshell_hist.ids) {
          free(tshell_hist.ids);
          free(tshell_hist.mat);
          free(tshell_hist.ndata);
          free(tshell_hist.nhist);
          free(tshell_hist.data);
          free(tshell_hist.hist);
          free(tshell_hist.strain);
        }
        have_tshell_hist = 0;
      }
    }
    if(have_beam) {
      if(! elout_beam(fp,handle,state,&beam)) {
        if(beam.ids) {
          free(beam.ids);
          free(beam.mat);
          free(beam.nip);
          free(beam.mtype);
          free(beam.axial);
          free(beam.shears);
          free(beam.sheart);
          free(beam.moments);
          free(beam.momentt);
          free(beam.torsion);
          free(beam.clength);
          free(beam.vforce);
          if(beam.s11) {
            free(beam.s11);
            free(beam.s12);
            free(beam.s31);
            free(beam.plastic);
          }
        }
        have_beam = 0;
      }
    } else if(have_beam_hist) {
      if(! elout_beam_hist(fp,handle,state,&beam_hist)) {
        if(beam_hist.ids) {
          free(beam_hist.ids);
          free(beam_hist.mat);
          free(beam_hist.ndata);
          free(beam_hist.nhist);
          free(beam_hist.data);
          free(beam_hist.hist);
          free(beam_hist.strain);
        }
        have_beam_hist = 0;
      }
    }
    if(have_shell) {
      if(! elout_shell(fp,handle,state,&shell)) {
        if(shell.ids) {
          free(shell.ids);
          free(shell.mat);
          free(shell.nip);
          free(shell.state);
          free(shell.iop);
          if(shell.npl) free(shell.npl);
          free(shell.damage);
          if(shell.lxx) {
            free(shell.lxx);
            free(shell.lyy);
            free(shell.lzz);
            free(shell.lxy);
            free(shell.lyz);
            free(shell.lzx);
          }
          if(shell.uxx) {
            free(shell.uxx);
            free(shell.uyy);
            free(shell.uzz);
            free(shell.uxy);
            free(shell.uyz);
            free(shell.uzx);
          }
          free(shell.sxx);
          free(shell.syy);
          free(shell.szz);
          free(shell.sxy);
          free(shell.syz);
          free(shell.szx);
          free(shell.ps);
        }
        have_shell=0;
      }
    } else if(have_shell_hist) {
      if(! elout_shell_hist(fp,handle,state,&shell_hist)) {
        if(shell_hist.ids) {
          free(shell_hist.ids);
          free(shell_hist.mat);
          free(shell_hist.ndata);
          free(shell_hist.nhist);
          free(shell_hist.data);
          free(shell_hist.hist);
          free(shell_hist.strain);
        }
        have_shell_hist = 0;
      }
    }
  }
  fclose(fp);
/*
  free everything here....
*/
  if(have_solid && solid.ids) {
    free(solid.ids);
    free(solid.mat);
    free(solid.state);
    free(solid.sxx);
    free(solid.syy);
    free(solid.szz);
    free(solid.sxy);
    free(solid.syz);
    free(solid.szx);
    free(solid.yield);
    free(solid.effsg);
  }
  if(have_beam && beam.ids) {
    free(beam.ids);
    free(beam.mat);
    free(beam.nip);
    free(beam.mtype);
    free(beam.axial);
    free(beam.shears);
    free(beam.sheart);
    free(beam.moments);
    free(beam.momentt);
    free(beam.torsion);
    free(beam.clength);
    free(beam.vforce);
    if(beam.s11) {
      free(beam.s11);
      free(beam.s12);
      free(beam.s31);
      free(beam.plastic);
    }
  }
  if(have_shell && shell.ids) {
    free(shell.ids);
    free(shell.mat);
    free(shell.nip);
    free(shell.state);
    free(shell.iop);
    free(shell.damage);
    if(shell.lxx) {
      free(shell.lxx);
      free(shell.lyy);
      free(shell.lzz);
      free(shell.lxy);
      free(shell.lyz);
      free(shell.lzx);
      free(shell.uxx);
      free(shell.uyy);
      free(shell.uzz);
      free(shell.uxy);
      free(shell.uyz);
      free(shell.uzx);
    }
    free(shell.sxx);
    free(shell.syy);
    free(shell.szz);
    free(shell.sxy);
    free(shell.syz);
    free(shell.szx);
    free(shell.ps);
  }
  if(have_solid_hist && solid_hist.ids) {
    free(solid_hist.ids);
    free(solid_hist.mat);
    free(solid_hist.ndata);
    free(solid_hist.nhist);
    free(solid_hist.data);
    free(solid_hist.hist);
    free(solid_hist.strain);
  }
  if(have_tshell_hist && tshell_hist.ids) {
    free(tshell_hist.ids);
    free(tshell_hist.mat);
    free(tshell_hist.ndata);
    free(tshell_hist.nhist);
    free(tshell_hist.data);
    free(tshell_hist.hist);
    free(tshell_hist.strain);
  }
  if(have_beam_hist && beam_hist.ids) {
    free(beam_hist.ids);
    free(beam_hist.mat);
    free(beam_hist.ndata);
    free(beam_hist.nhist);
    free(beam_hist.data);
    free(beam_hist.hist);
    free(beam_hist.strain);
  }
  if(have_shell_hist && shell_hist.ids) {
    free(shell_hist.ids);
    free(shell_hist.mat);
    free(shell_hist.ndata);
    free(shell_hist.nhist);
    free(shell_hist.data);
    free(shell_hist.hist);
    free(shell_hist.strain);
  }

  printf("      %d states extracted\n",state-1);
  return 0;
}

int
elout_solid(FILE *fp,int handle, int state, MDSOLID *solid)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length;
  int num;
  int have_strain,i;

  if(state<=999999)
    sprintf(dirname,"/elout/solid/d%6.6d",state);
  else
    sprintf(dirname,"/elout/solid/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"eps_xx",&typid,&length,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
  num=length;
  if(num > solid->idsize) {
    if(solid->ids) {
      free(solid->ids);
      free(solid->mat);
      free(solid->state);
      free(solid->sxx);
      free(solid->syy);
      free(solid->szz);
      free(solid->sxy);
      free(solid->syz);
      free(solid->szx);
      free(solid->yield);
      free(solid->effsg);
    }
    solid->ids = (int *) malloc(num*sizeof(int));
    solid->mat = (int *) malloc(num*sizeof(int));
    solid->state = (int *) malloc(num*sizeof(int));
    solid->sxx = (float *) malloc(num*sizeof(float));
    solid->syy = (float *) malloc(num*sizeof(float));
    solid->szz = (float *) malloc(num*sizeof(float));
    solid->sxy = (float *) malloc(num*sizeof(float));
    solid->syz = (float *) malloc(num*sizeof(float));
    solid->szx = (float *) malloc(num*sizeof(float));
    solid->yield = (float *) malloc(num*sizeof(float));
    solid->effsg = (float *) malloc(num*sizeof(float));
    solid->idsize = num;
  }
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,solid->ids) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"mtype",0,length,solid->mat) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"state",0,length,solid->state) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_xx",0,length,solid->sxx) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_yy",0,length,solid->syy) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_zz",0,length,solid->szz) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_xy",0,length,solid->sxy) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_yz",0,length,solid->syz) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_zx",0,length,solid->szx) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yield",0,length,solid->yield) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"effsg",0,length,solid->effsg) != length) return 0;
/*
  Output stress data
*/
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
  fprintf(fp," element  materl\n");
  fprintf(fp,"     ipt  stress       sig-xx      sig-yy      sig");
  fprintf(fp,"-zz      sig-xy      sig-yz      sig-zx           ");
  fprintf(fp,"            yield\n           state                    ");
  fprintf(fp,"                                                              ");
  fprintf(fp,"effsg      function\n");

  for(i=0; i<length; i++) {
    fprintf(fp,"%8d-%7d\n",solid->ids[i],solid->mat[i]);
    fprintf(fp,"       1 %-7s ",solid->states[solid->state[i]-1]);
    fprintf(fp,"%12.4E%12.4E%12.4E",solid->sxx[i],solid->syy[i],solid->szz[i]);
    fprintf(fp,"%12.4E%12.4E%12.4E%14.4E%14.4E\n",solid->sxy[i],solid->syz[i],
                                   solid->szx[i],solid->effsg[i],solid->yield[i]);
  }
  if(have_strain) {
    if(lsda_read(handle,LSDA_FLOAT,"eps_xx",0,length,solid->sxx) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yy",0,length,solid->syy) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zz",0,length,solid->szz) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_xy",0,length,solid->sxy) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yz",0,length,solid->syz) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zx",0,length,solid->szx) != length) return 0;

    fprintf(fp,"\n\n\n e l e m e n t   s t r a i n   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
    fprintf(fp," element  strain       eps-xx      eps-yy      eps");
    fprintf(fp,"-zz      eps-xy      eps-yz      eps-zx           ");
    fprintf(fp,"                 yield\n num/ipt   state\n");
/*
  yield?  But nothing is printed out there....ack!
*/
    for(i=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",solid->ids[i],solid->mat[i]);
      fprintf(fp,"      1  %-7s ",solid->states[solid->state[i]-1]);
      fprintf(fp,"%12.4E%12.4E%12.4E",solid->sxx[i],solid->syy[i],solid->szz[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",solid->sxy[i],solid->syz[i],solid->szx[i]);
    }
  }
  return 1;
}

int
elout_tshell(FILE *fp,int handle, int state, MDTSHELL *tshell)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length;
  int num,num1,num2;
  int have_strain,have_genoa,i,j,k;

  if(state<=999999)
    sprintf(dirname,"/elout/thickshell/d%6.6d",state);
  else
    sprintf(dirname,"/elout/thickshell/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"lower_eps_xx",&typid,&length,&filenum);
  num2=length;
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
  num=length;
  lsda_queryvar(handle,"genoa_damage",&typid,&length,&filenum);
  have_genoa= (typid > 0);
  lsda_queryvar(handle,"state",&typid,&length,&filenum);
  num1=length;
    tshell->ids = (int *) malloc(num*sizeof(int));
    tshell->mat = (int *) malloc(num*sizeof(int));
    tshell->state = (int *) malloc(num1*sizeof(int));
    tshell->nip = (int *) malloc(num*sizeof(int));
    tshell->damage = (int *) malloc(num1*sizeof(int));
    tshell->sxx = (float *) malloc(num1*sizeof(float));
    tshell->syy = (float *) malloc(num1*sizeof(float));
    tshell->szz = (float *) malloc(num1*sizeof(float));
    tshell->sxy = (float *) malloc(num1*sizeof(float));
    tshell->syz = (float *) malloc(num1*sizeof(float));
    tshell->szx = (float *) malloc(num1*sizeof(float));
    tshell->uxx = (float *) malloc(num2*sizeof(float));
    tshell->uyy = (float *) malloc(num2*sizeof(float));
    tshell->uzz = (float *) malloc(num2*sizeof(float));
    tshell->uxy = (float *) malloc(num2*sizeof(float));
    tshell->uyz = (float *) malloc(num2*sizeof(float));
    tshell->uzx = (float *) malloc(num2*sizeof(float));
    tshell->yield = (float *) malloc(num1*sizeof(float));
    tshell->effsg = (float *) malloc(num1*sizeof(float));
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,num,tshell->ids) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"mat",0,num,tshell->mat) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"nip",0,num,tshell->nip) != num) return 0;
  if(lsda_read(handle,LSDA_INT,"state",0,num1,tshell->state) != num1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_xx",0,num1,tshell->sxx) != num1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_yy",0,num1,tshell->syy) != num1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_zz",0,num1,tshell->szz) != num1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_xy",0,num1,tshell->sxy) != num1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_yz",0,num1,tshell->syz) != num1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_zx",0,num1,tshell->szx) != num1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yield",0,num1,tshell->yield) != num1) return 0;
  if(have_genoa)
    if(lsda_read(handle,LSDA_INT,"genoa_damage",0,num1,tshell->damage) != num1) return 0;
/*
  Output stress data
*/
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time%12.5E )\n\n",cycle,time);
  fprintf(fp," element  materl\n");
  fprintf(fp," num/ipt  stress       sig-xx      sig-yy      sig");
  fprintf(fp,"-zz      sig-xy      sig-yz      sig-zx         ");
  fprintf(fp,"yield%s\n           state                    ",(have_genoa?"     genoa":""));
  fprintf(fp,"                                                ");
  fprintf(fp,"           function%s\n",(have_genoa?"    damage":""));
  for(j=k=0;j<num;j++){
    fprintf(fp,"%8d-%5d\n",tshell->ids[j],tshell->mat[j]);
    for(i=0; i<tshell->nip[j]; i++,k++) {
      fprintf(fp,"%7d  %-7s ",i+1,tshell->states[tshell->state[k]-1]);
      fprintf(fp,"%12.4E%12.4E%12.4E",tshell->sxx[k],tshell->syy[k],tshell->szz[k]);
      fprintf(fp,"%12.4E%12.4E%12.4E%14.4E",tshell->sxy[k],tshell->syz[k],
                                  tshell->szx[k],tshell->yield[k]);
      if(have_genoa)
        if(tshell->damage[k]!=-1)fprintf(fp,"%10d",tshell->damage[k]);
      fprintf(fp,"\n");
    }
  }
  if(have_strain) {
    fprintf(fp,"\n strains (global)");
    fprintf(fp,"      eps-xx      eps-yy      eps");
    fprintf(fp,"-zz      eps-xy      eps-yz      eps-zx                                    \n ");
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_xx",0,num2,tshell->sxx) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_yy",0,num2,tshell->syy) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_zz",0,num2,tshell->szz) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_xy",0,num2,tshell->sxy) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_yz",0,num2,tshell->syz) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_zx",0,num2,tshell->szx) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_xx",0,num2,tshell->uxx) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_yy",0,num2,tshell->uyy) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_zz",0,num2,tshell->uzz) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_xy",0,num2,tshell->uxy) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_yz",0,num2,tshell->uyz) != num2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_zx",0,num2,tshell->uzx) != num2) return 0;
    for(i=0;i<num;i++){
      fprintf(fp,"\n%8d-%5d\n",tshell->ids[i],tshell->mat[i]);
      fprintf(fp," lower ipt       %12.4E%12.4E%12.4E",tshell->sxx[i],tshell->syy[i],tshell->szz[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",tshell->sxy[i],tshell->syz[i],tshell->szx[i]);
      fprintf(fp," upper ipt       %12.4E%12.4E%12.4E",tshell->uxx[i],tshell->uyy[i],tshell->uzz[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",tshell->uxy[i],tshell->uyz[i],tshell->uzx[i]);
    }
  }
      free(tshell->ids);
      free(tshell->mat);
      free(tshell->state);
      free(tshell->nip);
      free(tshell->damage);
      free(tshell->sxx);
      free(tshell->syy);
      free(tshell->szz);
      free(tshell->sxy);
      free(tshell->syz);
      free(tshell->szx);
      free(tshell->uxx);
      free(tshell->uyy);
      free(tshell->uzz);
      free(tshell->uxy);
      free(tshell->uyz);
      free(tshell->uzx);
      free(tshell->yield);
      free(tshell->effsg);
  return 1;
}

int
elout_beam(FILE *fp,int handle, int state, MDBEAM *beam)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length;
  int i,j,k,len2,have_mtype;

  if(state<=999999)
    sprintf(dirname,"/elout/beam/d%6.6d",state);
  else
    sprintf(dirname,"/elout/beam/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;
  if((int) length > beam->idsize) {
    if(beam->ids) {
      free(beam->ids);
      free(beam->mat);
      free(beam->nip);
      free(beam->mtype);
      free(beam->axial);
      free(beam->shears);
      free(beam->sheart);
      free(beam->moments);
      free(beam->momentt);
      free(beam->torsion);
      free(beam->clength);
      free(beam->vforce);
    }
    beam->ids = (int *) malloc(length*sizeof(int));
    beam->mat = (int *) malloc(length*sizeof(int));
    beam->nip = (int *) malloc(length*sizeof(int));
    beam->mtype = (int *) malloc(length*sizeof(int));
    beam->axial = (float *) malloc(length*sizeof(float));
    beam->shears = (float *) malloc(length*sizeof(float));
    beam->sheart = (float *) malloc(length*sizeof(float));
    beam->moments = (float *) malloc(length*sizeof(float));
    beam->momentt = (float *) malloc(length*sizeof(float));
    beam->torsion = (float *) malloc(length*sizeof(float));
    beam->clength = (float *) malloc(length*sizeof(float));
    beam->vforce = (float *) malloc(length*sizeof(float));
    beam->idsize = length;
  }
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,beam->ids) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"mat",0,length,beam->mat) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"nip",0,length,beam->nip) != length) return 0;
  have_mtype=1;
  if(lsda_read(handle,LSDA_INT,"mtype",0,length,beam->mtype) != length) have_mtype=0;
  for(i=len2=0; i < length; i++)
    len2 += beam->nip[i];
  if(len2 && len2 > beam->dsize) {
    if(beam->s11) {
      free(beam->s11);
      free(beam->s12);
      free(beam->s31);
      free(beam->plastic);
    }
    beam->s11 = (float *) malloc(len2*sizeof(float));
    beam->s12 = (float *) malloc(len2*sizeof(float));
    beam->s31 = (float *) malloc(len2*sizeof(float));
    beam->plastic = (float *) malloc(len2*sizeof(float));
    beam->dsize = len2;
  }
  if(lsda_read(handle,LSDA_FLOAT,"axial",0,length,beam->axial) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"shear_s",0,length,beam->shears) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"shear_t",0,length,beam->sheart) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"moment_s",0,length,beam->moments) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"moment_t",0,length,beam->momentt) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"torsion",0,length,beam->torsion) != length) return 0;
  if(have_mtype) {
    if(lsda_read(handle,LSDA_FLOAT,"coef_length",0,length,beam->clength) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"visc_force",0,length,beam->vforce) != length) return 0;
  }
  if(len2) {
    if(lsda_read(handle,LSDA_FLOAT,"sigma_11",0,len2,beam->s11) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sigma_12",0,len2,beam->s12) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"sigma_31",0,len2,beam->s31) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"plastic_eps",0,len2,beam->plastic) != len2) return 0;
  }
/*
  Output this data
*/
  fprintf(fp," r e s u l t a n t s   a n d   s t r e s s e s   ");
  fprintf(fp,"f o r   t i m e  s t e p%9d   ( at time%12.5E )\n",cycle,time);

  for(i=k=0; i<length; i++) {
    if(have_mtype) {
      fprintf(fp,"\n\n beam/truss # =%8d      part ID  =%10d      material type=%8d\n",
        beam->ids[i],beam->mat[i],beam->mtype[i]);
      if(beam->mtype[i] == 156) {
        fprintf(fp,"\n muscle data    length   actlevel  lgth ");
        fprintf(fp,"coef    epsrate   pass frc   visc frc  activ frc  total frc\n");
        fprintf(fp,"           %11.3E%11.3E%11.3E",beam->momentt[i],beam->moments[i],beam->clength[i]);
        fprintf(fp,"%11.3E%11.3E%11.3E%11.3E%11.3E\n",beam->torsion[i],beam->shears[i],beam->vforce[i],beam->sheart[i],beam->axial[i]);
      } else {
        fprintf(fp,"\n resultants      axial    shear-s    she");
        fprintf(fp,"ar-t    moment-s   moment-t   torsion\n");
        fprintf(fp,"           %11.3E%11.3E%11.3E",beam->axial[i],beam->shears[i],beam->sheart[i]);
        fprintf(fp,"%11.3E%11.3E%11.3E\n",beam->moments[i],beam->momentt[i],beam->torsion[i]);
      }
    } else {
      fprintf(fp,"\n\n\n\n beam/truss # =%8d      material #  =%5d\n",
        beam->ids[i],beam->mat[i]);
      fprintf(fp,"\n\n resultants      axial    shear-s    she");
      fprintf(fp,"ar-t    moment-s   moment-t   torsion\n");
      fprintf(fp,"           %11.3E%11.3E%11.3E",beam->axial[i],beam->shears[i],beam->sheart[i]);
      fprintf(fp,"%11.3E%11.3E%11.3E\n",beam->moments[i],beam->momentt[i],beam->torsion[i]);
    }
    if(beam->nip[i]) {
      fprintf(fp,"\n\n integration point stresses\n");
      fprintf(fp,"                       ");
      fprintf(fp,"sigma 11       sigma 12       sigma 31   plastic  eps\n");
      for(j=0; j<beam->nip[i]; j++,k++)
        fprintf(fp,"%5d           %15.6E%15.6E%15.6E%15.6E\n",j+1,
         beam->s11[k],beam->s12[k],beam->s31[k],beam->plastic[k]);
    }
  }
  return 1;
}

int
elout_shell(FILE *fp,int handle, int state, MDSHELL *shell)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length,length2;
  int have_strain,have_genoa,i,j,k,len2;
  int jj,k1,k2;
  int have_npl;

  if(state<=999999)
    sprintf(dirname,"/elout/shell/d%6.6d",state);
  else
    sprintf(dirname,"/elout/shell/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"npl",&typid,&length2,&filenum);
  have_npl = (typid > 0);
  if(have_npl) {
    lsda_queryvar(handle,"eps_xx",&typid,&length2,&filenum);
    have_strain = (typid > 0);
  } else {
    lsda_queryvar(handle,"lower_eps_xx",&typid,&length2,&filenum);
    have_strain = (typid > 0);
  }
  lsda_queryvar(handle,"genoa_damage",&typid,&length2,&filenum);
  have_genoa = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;
  if((int) length > shell->idsize) {
    if(shell->ids) {
      free(shell->ids);
      free(shell->mat);
      free(shell->nip);
      free(shell->iop);
      if(shell->npl) free(shell->npl);
      if(shell->lxx) {
        free(shell->lxx);
        free(shell->lyy);
        free(shell->lzz);
        free(shell->lxy);
        free(shell->lyz);
        free(shell->lzx);
      }
      if(shell->uxx) {
        free(shell->uxx);
        free(shell->uyy);
        free(shell->uzz);
        free(shell->uxy);
        free(shell->uyz);
        free(shell->uzx);
      }
    }
    shell->ids = (int *) malloc(length*sizeof(int));
    shell->mat = (int *) malloc(length*sizeof(int));
    shell->nip = (int *) malloc(length*sizeof(int));
    shell->iop = (int *) malloc(length*sizeof(int));
    if(have_npl) {
      shell->npl = (int *) malloc(length*sizeof(int));
/* don't know how big lxx needs to be yet, so allocate it below */
      shell->uxx = NULL;
    } else {
      shell->npl = NULL;
      if(have_strain) {
        shell->lxx = (float *) malloc(length*sizeof(float));
        shell->lyy = (float *) malloc(length*sizeof(float));
        shell->lzz = (float *) malloc(length*sizeof(float));
        shell->lxy = (float *) malloc(length*sizeof(float));
        shell->lyz = (float *) malloc(length*sizeof(float));
        shell->lzx = (float *) malloc(length*sizeof(float));
        shell->uxx = (float *) malloc(length*sizeof(float));
        shell->uyy = (float *) malloc(length*sizeof(float));
        shell->uzz = (float *) malloc(length*sizeof(float));
        shell->uxy = (float *) malloc(length*sizeof(float));
        shell->uyz = (float *) malloc(length*sizeof(float));
        shell->uzx = (float *) malloc(length*sizeof(float));
      }
    }
    shell->idsize = length;
  }
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,shell->ids) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"mat",0,length,shell->mat) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"nip",0,length,shell->nip) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"iop",0,length,shell->iop) != length) return 0;
  if(have_npl) {
    if(lsda_read(handle,LSDA_INT,"npl",0,length,shell->npl) != length) return 0;
    for(i=len2=0; i < length; i++)
      len2 += shell->nip[i]*shell->npl[i];
    if(have_strain) {
      shell->lxx = (float *) malloc(len2*sizeof(float));
      shell->lyy = (float *) malloc(len2*sizeof(float));
      shell->lzz = (float *) malloc(len2*sizeof(float));
      shell->lxy = (float *) malloc(len2*sizeof(float));
      shell->lyz = (float *) malloc(len2*sizeof(float));
      shell->lzx = (float *) malloc(len2*sizeof(float));
    }
  } else {
    for(i=len2=0; i < length; i++)
      len2 += shell->nip[i];
  }
  if(len2 > shell->dsize) {
    if(shell->sxx) {
      free(shell->state);
      free(shell->sxx);
      free(shell->syy);
      free(shell->szz);
      free(shell->sxy);
      free(shell->syz);
      free(shell->szx);
      free(shell->ps);
      free(shell->damage);
    }
    shell->state = (int *) malloc(len2*sizeof(int));
    shell->sxx = (float *) malloc(len2*sizeof(float));
    shell->syy = (float *) malloc(len2*sizeof(float));
    shell->szz = (float *) malloc(len2*sizeof(float));
    shell->sxy = (float *) malloc(len2*sizeof(float));
    shell->syz = (float *) malloc(len2*sizeof(float));
    shell->szx = (float *) malloc(len2*sizeof(float));
    shell->ps  = (float *) malloc(len2*sizeof(float));
    shell->damage= (int *) malloc(len2*sizeof(int));
    shell->dsize = len2;
  }
  if(lsda_read(handle,LSDA_INT,"state",0,len2,shell->state) != len2) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_xx",0,len2,shell->sxx) != len2) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_yy",0,len2,shell->syy) != len2) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_zz",0,len2,shell->szz) != len2) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_xy",0,len2,shell->sxy) != len2) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_yz",0,len2,shell->syz) != len2) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"sig_zx",0,len2,shell->szx) != len2) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"plastic_strain",0,len2,shell->ps) != len2) return 0;
  if(have_npl && have_strain) {
    if(lsda_read(handle,LSDA_FLOAT,"eps_xx",0,len2,shell->lxx) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yy",0,len2,shell->lyy) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zz",0,len2,shell->lzz) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_xy",0,len2,shell->lxy) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yz",0,len2,shell->lyz) != len2) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zx",0,len2,shell->lzx) != len2) return 0;
  }
  if(have_genoa)
    if(lsda_read(handle,LSDA_INT,"genoa_damage",0,len2,shell->damage) != len2) return 0;
/*
  Output this data
*/
  if(have_npl) {
    fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
    sprintf(dirname,"(%s)",shell->system);
    fprintf(fp," element  materl%-8s\n",dirname);
    fprintf(fp," ipt-shl component         xx          yy         ");
    fprintf(fp," zz          xy          yz          zx       plastic%s\n",(have_genoa?"     genoa":""));
    fprintf(fp,"                                                  ");
    fprintf(fp,"                                               strain %s\n",(have_genoa?"   damage":""));

    for(i=k1=k2=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",shell->ids[i],shell->mat[i]);
      for(j=0; j<shell->nip[i]; j++) {
        for(jj=0; jj<shell->npl[i]; jj++,k1++) {
          fprintf(fp,"%4d  p%1d %-7s ",j+1,jj+1,"stress");
          fprintf(fp,"%12.4E%12.4E%12.4E",shell->sxx[k1],shell->syy[k1],shell->szz[k1]);
          fprintf(fp,"%12.4E%12.4E%12.4E%14.4E",shell->sxy[k1],shell->syz[k1],
                                                  shell->szx[k1],shell->ps[k1]);
          if(have_genoa)
            if(shell->damage[k1]!=-1)fprintf(fp,"%10d",shell->damage[k1]);
          fprintf(fp,"\n");
        }
        if(have_strain) {
          for(jj=0; jj<shell->npl[i]; jj++,k2++) {
            fprintf(fp,"%4d  p%1d %-7s ",j+1,jj+1,"strain");
            fprintf(fp,"%12.4E%12.4E%12.4E"  ,shell->lxx[k2],shell->lyy[k2],shell->lzz[k2]);
            fprintf(fp,"%12.4E%12.4E%12.4E\n",shell->lxy[k2],shell->lyz[k2],shell->lzx[k2]);
          }
        }
      }
    }
  } else {
    fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n\n",cycle,time);
    sprintf(dirname,"(%s)",shell->system);
    fprintf(fp," element  materl%-8s\n",dirname);
    fprintf(fp," ipt-shl  stress       sig-xx      sig-yy      sig");
    fprintf(fp,"-zz      sig-xy      sig-yz      sig-zx       plastic%s\n",(have_genoa?"     genoa":""));
    fprintf(fp,"           state                                  ");
    fprintf(fp,"                                               strain %s\n",(have_genoa?"   damage":""));

    for(i=k=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",shell->ids[i],shell->mat[i]);
      for(j=0; j<shell->nip[i]; j++,k++) {
        fprintf(fp,"%4d-%3d %-7s ",j+1,shell->iop[i],shell->states[shell->state[k]-1]);
        fprintf(fp,"%12.4E%12.4E%12.4E",shell->sxx[k],shell->syy[k],shell->szz[k]);
        fprintf(fp,"%12.4E%12.4E%12.4E%14.4E",shell->sxy[k],shell->syz[k],
                                              shell->szx[k],shell->ps[k]);
        if(have_genoa)
          if(shell->damage[k]!=-1)fprintf(fp,"%10d",shell->damage[k]);
        fprintf(fp,"\n");
      }
    }
  }
  if(have_strain && !have_npl) {
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_xx",0,length,shell->lxx) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_yy",0,length,shell->lyy) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_zz",0,length,shell->lzz) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_xy",0,length,shell->lxy) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_yz",0,length,shell->lyz) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"lower_eps_zx",0,length,shell->lzx) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_xx",0,length,shell->uxx) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_yy",0,length,shell->uyy) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_zz",0,length,shell->uzz) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_xy",0,length,shell->uxy) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_yz",0,length,shell->uyz) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"upper_eps_zx",0,length,shell->uzx) != length) return 0;

/*
  all the blanks here are silly, but are output by DYNA so.....
*/
    sprintf(dirname,"(%s)",shell->system);
    fprintf(fp,"\n strains %-8s      eps-xx      eps-yy      eps",dirname);
    fprintf(fp,"-zz      eps-xy      eps-yz      eps-zx                                    \n");
    fprintf(fp,"                                                  ");
    fprintf(fp,"                                                                   \n");

    for(i=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",shell->ids[i],shell->mat[i]);
      fprintf(fp," lower ipt       %12.4E%12.4E%12.4E",shell->lxx[i],shell->lyy[i],shell->lzz[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",shell->lxy[i],shell->lyz[i],shell->lzx[i]);
      fprintf(fp," upper ipt       %12.4E%12.4E%12.4E",shell->uxx[i],shell->uyy[i],shell->uzz[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",shell->uxy[i],shell->uyz[i],shell->uzx[i]);
    }
  }
  return 1;
}
int
elout_solid_hist(FILE *fp,int handle, int state, MDHIST *dp)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length,lhist,lstr,ldata;
  int num;
  int have_strain,have_hist,i,j,k,kk,l,n,nips,istate,perip,nlines;

  if(state<=999999)
    sprintf(dirname,"/elout/solid_hist/d%6.6d",state);
  else
    sprintf(dirname,"/elout/solid_hist/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"data",&typid,&ldata,&filenum);
  lsda_queryvar(handle,"hist",&typid,&lhist,&filenum);
  have_hist = (typid > 0);
  lsda_queryvar(handle,"strain",&typid,&lstr,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
  num=length;
  if(num > dp->idsize) {
    if(dp->ids) {
      free(dp->ids);
      free(dp->mat);
      free(dp->ndata);
      free(dp->data);
      free(dp->nhist);
      free(dp->hist);
      free(dp->strain);
    }
    dp->ids = (int *) malloc(num*sizeof(int));
    dp->mat = (int *) malloc(num*sizeof(int));
    dp->ndata = (int *) malloc(num*sizeof(int));
    dp->data = (float *) malloc(ldata*sizeof(float));
    if(have_hist) {
      dp->nhist = (int *) malloc(num*sizeof(int));
      dp->hist = (float *) malloc(lhist*sizeof(float));
    } else {
      dp->nhist = (int *) malloc(sizeof(int));
      dp->hist = (float *) malloc(sizeof(float));
    }
    if(have_strain) {
      dp->strain = (float *) malloc(lstr*sizeof(float));
    } else {
      dp->strain = (float *) malloc(sizeof(float));
    }
  }
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"nhv",0,1,&dp->nhv) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,dp->ids) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"mats",0,length,dp->mat) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"ndata",0,length,dp->ndata) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"data",0,ldata,dp->data) != ldata) return 0;
  if(have_hist) {
    if(lsda_read(handle,LSDA_INT,"nhist",0,length,dp->nhist) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"hist",0,lhist,dp->hist) != lhist) return 0;
  }
  if(have_strain) {
    if(lsda_read(handle,LSDA_FLOAT,"strain",0,lstr,dp->strain) != lstr) return 0;
  }
/*
  Output stress data
*/
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
  fprintf(fp," element part ID   (solid)\n");
  fprintf(fp,"     ipt  stress    stress-xx   stress-yy   stress");
  fprintf(fp,"-zz   stress-xy   stress-yz   stress-zx           ");
  fprintf(fp,"            yield\n           state                    ");
  fprintf(fp,"                                                              ");
  fprintf(fp,"effsg      function\n");

  for(i=k=0; i<length; i++) {
    fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
    nips = dp->ndata[i]/9;
    for(j=0; j<nips; j++) {
      istate = dp->data[k];
      fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
      fprintf(fp,"%12.4E%12.4E%12.4E",dp->data[k+1],dp->data[k+2],
                                      dp->data[k+3]);
      fprintf(fp,"%12.4E%12.4E%12.4E%14.4E%14.4E\n",dp->data[k+4],dp->data[k+5],
                                   dp->data[k+6],dp->data[k+7],dp->data[k+8]);
      k=k+9;
    }
  }
  if(have_hist) {
    fprintf(fp,"\n\n\n e l e m e n t   h i s t r y   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (solid)\n");
    fprintf(fp,"     ipt            history 1   history 2   history 3   history 4");
    fprintf(fp,"   history 5   history 6   history 7   history 8\n");
    nlines = (dp->nhv-1)/8+1;
    for(i=k=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      nips = dp->ndata[i]/9;
      perip = dp->nhist[i]/nips;
  /* This part is a bit strange.  The ascii routines print out nhv entries,
     8 per line.  But there may be fewer data values than that, in which case
     it just prints out blanks until nhv entries have been output */
      for(j=0; j<nips; j++) {
        istate = dp->hist[k];
        for(l=0,kk=1; l<nlines; l++,kk+=8) {
          fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
          for(n=kk; n<kk+8 && n<perip; n++)
            fprintf(fp,"%12.4E",dp->hist[k+n]);
          fprintf(fp,"\n");
        }
        k += perip;
      }
    }
  }
  if(have_strain) {
  /* strain is easy -- for solids there are 6 words per element */

    fprintf(fp,"\n\n\n e l e m e n t   s t r a i n   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (solid)\n");
    fprintf(fp,"                    strain-xx   strain-yy   strain-zz   strain-xy   strain-yz   strain-zx\n");
    for(i=0; i<length; i++) {
      j=6*i;
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      fprintf(fp,"         average ");
      fprintf(fp,"%12.4E%12.4E%12.4E"  ,dp->strain[j  ],dp->strain[j+1],dp->strain[j+2]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",dp->strain[j+3],dp->strain[j+4],dp->strain[j+5]);
    }
  }
  return 1;
}
int
elout_tshell_hist(FILE *fp,int handle, int state, MDHIST *dp)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length,lhist,lstr,ldata;
  int num;
  int have_strain,have_hist,i,j,k,kk,l,n,nips,istate,perip,nlines;

  if(state<=999999)
    sprintf(dirname,"/elout/thickshell_hist/d%6.6d",state);
  else
    sprintf(dirname,"/elout/thickshell_hist/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"data",&typid,&ldata,&filenum);
  lsda_queryvar(handle,"hist",&typid,&lhist,&filenum);
  have_hist = (typid > 0);
  lsda_queryvar(handle,"strain",&typid,&lstr,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
  num=length;
  if(num > dp->idsize) {
    if(dp->ids) {
      free(dp->ids);
      free(dp->mat);
      free(dp->ndata);
      free(dp->data);
      free(dp->nhist);
      free(dp->hist);
      free(dp->strain);
    }
    dp->ids = (int *) malloc(num*sizeof(int));
    dp->mat = (int *) malloc(num*sizeof(int));
    dp->ndata = (int *) malloc(num*sizeof(int));
    dp->data = (float *) malloc(ldata*sizeof(float));
    if(have_hist) {
      dp->nhist = (int *) malloc(num*sizeof(int));
      dp->hist = (float *) malloc(lhist*sizeof(float));
    } else {
      dp->nhist = (int *) malloc(sizeof(int));
      dp->hist = (float *) malloc(sizeof(float));
    }
    if(have_strain) {
      dp->strain = (float *) malloc(lstr*sizeof(float));
    } else {
      dp->strain = (float *) malloc(sizeof(float));
    }
  }
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"nhv",0,1,&dp->nhv) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,dp->ids) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"mats",0,length,dp->mat) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"ndata",0,length,dp->ndata) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"data",0,ldata,dp->data) != ldata) return 0;
  if(have_hist) {
    if(lsda_read(handle,LSDA_INT,"nhist",0,length,dp->nhist) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"hist",0,lhist,dp->hist) != lhist) return 0;
  }
  if(have_strain) {
    if(lsda_read(handle,LSDA_FLOAT,"strain",0,lstr,dp->strain) != lstr) return 0;
  }
/*
  Output stress data
*/
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
  fprintf(fp," element part ID   (thick shell)\n");
  fprintf(fp,"     ipt  stress    stress-xx   stress-yy   stress");
  fprintf(fp,"-zz   stress-xy   stress-yz   stress-zx           ");
  fprintf(fp,"            yield\n           state                    ");
  fprintf(fp,"                                                              ");
  fprintf(fp,"effsg      function\n");

  for(i=k=0; i<length; i++) {
    fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
    nips = dp->ndata[i]/9;
    for(j=0; j<nips; j++) {
      istate = dp->data[k];
      fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
      fprintf(fp,"%12.4E%12.4E%12.4E",dp->data[k+1],dp->data[k+2],
                                      dp->data[k+3]);
      fprintf(fp,"%12.4E%12.4E%12.4E%14.4E%14.4E\n",dp->data[k+4],dp->data[k+5],
                                   dp->data[k+6],dp->data[k+7],dp->data[k+8]);
      k=k+9;
    }
  }
  if(have_hist) {
    fprintf(fp,"\n\n\n e l e m e n t   h i s t r y   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (thick shell)\n");
    fprintf(fp,"     ipt            history 1   history 2   history 3   history 4");
    fprintf(fp,"   history 5   history 6   history 7   history 8\n");
    nlines = (dp->nhv-1)/8+1;
    for(i=k=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      nips = dp->ndata[i]/9;
      perip = dp->nhist[i]/nips;
  /* This part is a bit strange.  The ascii routines print out nhv entries,
     8 per line.  But there may be fewer data values than that, in which case
     it just prints out blanks until nhv entries have been output */
      for(j=0; j<nips; j++) {
        istate = dp->hist[k];
        for(l=0,kk=1; l<nlines; l++,kk+=8) {
          fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
          for(n=kk; n<kk+8 && n<perip; n++)
            fprintf(fp,"%12.4E",dp->hist[k+n]);
          fprintf(fp,"\n");
        }
        k += perip;
      }
    }
  }
  if(have_strain) {
  /* strain is easy -- for solids there are 6 words per element */

    fprintf(fp,"\n\n\n e l e m e n t   s t r a i n   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (thick shell)\n");
    fprintf(fp,"                    strain-xx   strain-yy   strain-zz   strain-xy   strain-yz   strain-zx\n");
    for(i=0; i<length; i++) {
      j=12*i;
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      fprintf(fp,"   lower ipt     ");
      fprintf(fp,"%12.4E%12.4E%12.4E"  ,dp->strain[j  ],dp->strain[j+1],dp->strain[j+2]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",dp->strain[j+3],dp->strain[j+4],dp->strain[j+5]);
      j += 6;
      fprintf(fp,"   upper ipt     ");
      fprintf(fp,"%12.4E%12.4E%12.4E"  ,dp->strain[j  ],dp->strain[j+1],dp->strain[j+2]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",dp->strain[j+3],dp->strain[j+4],dp->strain[j+5]);
    }
  }
  return 1;
}
int
elout_beam_hist(FILE *fp,int handle, int state, MDHIST *dp)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length,lhist,lstr,ldata;
  int num;
  int have_strain,have_hist,i,j,k,kk,l,n,nips,istate,perip,nlines;

  if(state<=999999)
    sprintf(dirname,"/elout/beam_hist/d%6.6d",state);
  else
    sprintf(dirname,"/elout/beam_hist/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"data",&typid,&ldata,&filenum);
  lsda_queryvar(handle,"hist",&typid,&lhist,&filenum);
  have_hist = (typid > 0);
  lsda_queryvar(handle,"strain",&typid,&lstr,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
  num=length;
  if(num > dp->idsize) {
    if(dp->ids) {
      free(dp->ids);
      free(dp->mat);
      free(dp->ndata);
      free(dp->data);
      free(dp->nhist);
      free(dp->hist);
      free(dp->strain);
    }
    dp->ids = (int *) malloc(num*sizeof(int));
    dp->mat = (int *) malloc(num*sizeof(int));
    dp->ndata = (int *) malloc(num*sizeof(int));
    dp->data = (float *) malloc(ldata*sizeof(float));
    if(have_hist) {
      dp->nhist = (int *) malloc(num*sizeof(int));
      dp->hist = (float *) malloc(lhist*sizeof(float));
    } else {
      dp->nhist = (int *) malloc(sizeof(int));
      dp->hist = (float *) malloc(sizeof(float));
    }
    if(have_strain) {
      dp->strain = (float *) malloc(lstr*sizeof(float));
    } else {
      dp->strain = (float *) malloc(sizeof(float));
    }
  }
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"nhv",0,1,&dp->nhv) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,dp->ids) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"mats",0,length,dp->mat) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"ndata",0,length,dp->ndata) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"data",0,ldata,dp->data) != ldata) return 0;
  if(have_hist) {
    if(lsda_read(handle,LSDA_INT,"nhist",0,length,dp->nhist) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"hist",0,lhist,dp->hist) != lhist) return 0;
  }
  if(have_strain) {
    if(lsda_read(handle,LSDA_FLOAT,"strain",0,lstr,dp->strain) != lstr) return 0;
  }
/*
  Output stress data
*/
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
  fprintf(fp," element part ID   (beam)\n");
  fprintf(fp,"     ipt  stress    stress-xx   stress-yy   stress");
  fprintf(fp,"-zz   stress-xy   stress-yz   stress-zx           ");
  fprintf(fp,"            yield\n           state                    ");
  fprintf(fp,"                                                              ");
  fprintf(fp,"effsg      function\n");

  for(i=k=0; i<length; i++) {
    fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
    nips = dp->ndata[i]/9;
    for(j=0; j<nips; j++) {
      istate = dp->data[k];
      fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
      fprintf(fp,"%12.4E%12.4E%12.4E",dp->data[k+1],dp->data[k+2],
                                      dp->data[k+3]);
      fprintf(fp,"%12.4E%12.4E%12.4E%14.4E%14.4E\n",dp->data[k+4],dp->data[k+5],
                                   dp->data[k+6],dp->data[k+7],dp->data[k+8]);
      k=k+9;
    }
  }
  if(have_hist) {
    fprintf(fp,"\n\n\n e l e m e n t   h i s t r y   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (beam)\n");
    fprintf(fp,"     ipt            history 1   history 2   history 3   history 4");
    fprintf(fp,"   history 5   history 6   history 7   history 8\n");
    nlines = (dp->nhv-1)/8+1;
    for(i=k=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      nips = dp->ndata[i]/9;
      perip = dp->nhist[i]/nips;
  /* This part is a bit strange.  The ascii routines print out nhv entries,
     8 per line.  But there may be fewer data values than that, in which case
     it just prints out blanks until nhv entries have been output */
      for(j=0; j<nips; j++) {
        istate = dp->hist[k];
        for(l=0,kk=1; l<nlines; l++,kk+=8) {
          fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
          for(n=kk; n<kk+8 && n<perip; n++)
            fprintf(fp,"%12.4E",dp->hist[k+n]);
          fprintf(fp,"\n");
        }
        k += perip;
      }
    }
  }
  if(have_strain) {
  /* strain is easy -- for solids there are 6 words per element */

    fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   r e s u l t a n t s    ");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (beam)\n");
    fprintf(fp,"                        axial     shear-s     shear-t    moment-s    moment-t      torsion\n");
    for(i=0; i<length; i++) {
      j=6*i;
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      fprintf(fp,"      resultants ");
      fprintf(fp,"%12.4E%12.4E%12.4E"  ,dp->strain[j  ],dp->strain[j+1],dp->strain[j+2]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",dp->strain[j+3],dp->strain[j+4],dp->strain[j+5]);
    }
  }
  return 1;
}
int
elout_shell_hist(FILE *fp,int handle, int state, MDHIST *dp)
{
  char dirname[128];
  float time;
  int cycle;
  int typid, filenum;
  LSDA_Length length,lhist,lstr,ldata;
  int num;
  int have_strain,have_hist,i,j,k,kk,l,n,nips,istate,perip,nlines;

  if(state<=999999)
    sprintf(dirname,"/elout/shell_hist/d%6.6d",state);
  else
    sprintf(dirname,"/elout/shell_hist/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"data",&typid,&ldata,&filenum);
  lsda_queryvar(handle,"hist",&typid,&lhist,&filenum);
  have_hist = (typid > 0);
  lsda_queryvar(handle,"strain",&typid,&lstr,&filenum);
  have_strain = (typid > 0);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;  /* all elements deleted */
  num=length;
  if(num > dp->idsize) {
    if(dp->ids) {
      free(dp->ids);
      free(dp->mat);
      free(dp->ndata);
      free(dp->data);
      free(dp->nhist);
      free(dp->hist);
      free(dp->strain);
    }
    dp->ids = (int *) malloc(num*sizeof(int));
    dp->mat = (int *) malloc(num*sizeof(int));
    dp->ndata = (int *) malloc(num*sizeof(int));
    dp->data = (float *) malloc(ldata*sizeof(float));
    if(have_hist) {
      dp->nhist = (int *) malloc(num*sizeof(int));
      dp->hist = (float *) malloc(lhist*sizeof(float));
    } else {
      dp->nhist = (int *) malloc(sizeof(int));
      dp->hist = (float *) malloc(sizeof(float));
    }
    if(have_strain) {
      dp->strain = (float *) malloc(lstr*sizeof(float));
    } else {
      dp->strain = (float *) malloc(sizeof(float));
    }
  }
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"nhv",0,1,&dp->nhv) != 1) return 0;
  if(lsda_read(handle,LSDA_INT,"ids",0,length,dp->ids) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"mats",0,length,dp->mat) != length) return 0;
  if(lsda_read(handle,LSDA_INT,"ndata",0,length,dp->ndata) != length) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"data",0,ldata,dp->data) != ldata) return 0;
  if(have_hist) {
    if(lsda_read(handle,LSDA_INT,"nhist",0,length,dp->nhist) != length) return 0;
    if(lsda_read(handle,LSDA_FLOAT,"hist",0,lhist,dp->hist) != lhist) return 0;
  }
  if(have_strain) {
    if(lsda_read(handle,LSDA_FLOAT,"strain",0,lstr,dp->strain) != lstr) return 0;
  }
/*
  Output stress data
*/
  fprintf(fp,"\n\n\n e l e m e n t   s t r e s s   c a l c u l a t i o n s");
  fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
  fprintf(fp," element part ID   (shell)\n");
  fprintf(fp,"     ipt  stress    stress-xx   stress-yy   stress");
  fprintf(fp,"-zz   stress-xy   stress-yz   stress-zx           ");
  fprintf(fp,"            yield\n           state                    ");
  fprintf(fp,"                                                              ");
  fprintf(fp,"effsg      function\n");

  for(i=k=0; i<length; i++) {
    fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
    nips = dp->ndata[i]/9;
    for(j=0; j<nips; j++) {
      istate = dp->data[k];
      fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
      fprintf(fp,"%12.4E%12.4E%12.4E",dp->data[k+1],dp->data[k+2],
                                      dp->data[k+3]);
      fprintf(fp,"%12.4E%12.4E%12.4E%14.4E%14.4E\n",dp->data[k+4],dp->data[k+5],
                                   dp->data[k+6],dp->data[k+7],dp->data[k+8]);
      k=k+9;
    }
  }
  if(have_hist) {
    fprintf(fp,"\n\n\n e l e m e n t   h i s t r y   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (shell)\n");
    fprintf(fp,"     ipt            history 1   history 2   history 3   history 4");
    fprintf(fp,"   history 5   history 6   history 7   history 8\n");
    nlines = (dp->nhv-1)/8+1;
    for(i=k=0; i<length; i++) {
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      nips = dp->ndata[i]/9;
      perip = dp->nhist[i]/nips;
  /* This part is a bit strange.  The ascii routines print out nhv entries,
     8 per line.  But there may be fewer data values than that, in which case
     it just prints out blanks until nhv entries have been output */
      for(j=0; j<nips; j++) {
        istate = dp->hist[k];
        for(l=0,kk=1; l<nlines; l++,kk+=8) {
          fprintf(fp,"%8d %-7s ",j+1,dp->states[istate-1]);
          for(n=kk; n<kk+8 && n<perip; n++)
            fprintf(fp,"%12.4E",dp->hist[k+n]);
          fprintf(fp,"\n");
        }
        k += perip;
      }
    }
  }
  if(have_strain) {
  /* strain is easy -- for solids there are 6 words per element */

    fprintf(fp,"\n\n\n e l e m e n t   s t r a i n   c a l c u l a t i o n s");
    fprintf(fp,"   f o r   t i m e  s t e p%9d   ( at time %12.5E )\n",cycle,time);
    fprintf(fp," element part ID   (shell)\n");
    fprintf(fp,"                    strain-xx   strain-yy   strain-zz   strain-xy   strain-yz   strain-zx\n");
    for(i=0; i<length; i++) {
      j=12*i;
      fprintf(fp,"%8d-%7d\n",dp->ids[i],dp->mat[i]);
      fprintf(fp,"   lower ipt     ");
      fprintf(fp,"%12.4E%12.4E%12.4E"  ,dp->strain[j  ],dp->strain[j+1],dp->strain[j+2]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",dp->strain[j+3],dp->strain[j+4],dp->strain[j+5]);
      j += 6;
      fprintf(fp,"   upper ipt     ");
      fprintf(fp,"%12.4E%12.4E%12.4E"  ,dp->strain[j  ],dp->strain[j+1],dp->strain[j+2]);
      fprintf(fp,"%12.4E%12.4E%12.4E\n",dp->strain[j+3],dp->strain[j+4],dp->strain[j+5]);
    }
  }
  return 1;
}
/*
  GLSTAT file
*/
int translate_glstat(int handle)
{
  int i,j,k,typid,filenum,state,part;
  LSDA_Length length;
  char dirname[640];
  char types[40][16];
  int cycle;
  float *swe;
  int nsw;
  float x;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/glstat/metadata") == -1) return 0;
  printf("Extracting GLSTAT data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"element_types",&typid,&length,&filenum);
  lsda_read(handle,LSDA_I1,"element_types",0,length,dirname);
/*
  parse dirname to split out the element types
  zero all pointers first to prevent garbage output
*/
  for(i=0; i<40; i++)
    types[i][0]=0;
  for(i=j=k=0; i<length; i++) {
    if(dirname[i] == ',') {
      types[j][k]=0;
      j++;
      k=0;
    } else {
      types[j][k++]=dirname[i];
    }
  }
  types[j][k]=0;
/*
  see if we have any stonewall energies, and if so allocate space for
  them
*/
  lsda_queryvar(handle,"../d000001/stonewall_energy",&typid,&length,&filenum);
  if(typid > 0) {
    nsw = length;
    swe = (float *) malloc(nsw*sizeof(float));
  } else {
    nsw = -1;
  }
/*
  open file and write header
*/
  sprintf(output_file,"%sglstat",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/glstat/metadata",fp);
/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/glstat",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_INT,"ts_eltype",0,1,&i) != 1) {
/* older codes didn't write ts_eltype if there was a loadcurve */
      fprintf(fp,"\n\n dt of cycle%8d is controlled by load curve\n\n",cycle);
    } else {
      if(lsda_read(handle,LSDA_INT,"ts_element",0,1,&j) != 1) break;
      part=0;
/* newer codes always write ts_part, some didn't but we'll just ignore the error and
   part will be 0 which is fine */
      lsda_read(handle,LSDA_INT,"ts_part",0,1,&part);
      if(i == 14 || i == 19) {
        fprintf(fp,"\n\n dt of cycle%8d is controlled by %s\n",cycle,types[i-1]);
      } else if(part == 0) {
        fprintf(fp,"\n\n dt of cycle%8d is controlled by %s%10d\n",
        cycle,types[i-1],j);
      } else {
        fprintf(fp,"\n\n dt of cycle%8d is controlled by %s%10d of part%10d\n",
        cycle,types[i-1],j,part);

      }
    }
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&x) != 1) break;
    fprintf(fp," time...........................%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"time_step",0,1,&x) != 1) break;
    fprintf(fp," time step......................%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"kinetic_energy",0,1,&x) != 1) break;
    fprintf(fp," kinetic energy.................%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,1,&x) != 1) break;
    fprintf(fp," internal energy................%14.5E\n",x);
    if(nsw > 0) {
      if(lsda_read(handle,LSDA_FLOAT,"stonewall_energy",0,nsw,swe) != nsw) break;
      for(i=0; i<nsw; i++)
        fprintf(fp," stonewall energy...............%14.5E wall#%2d\n",swe[i],i+1);
    }
    if(lsda_read(handle,LSDA_FLOAT,"rb_stopper_energy",0,1,&x) == 1)
      fprintf(fp," rigid body stopper energy......%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"spring_and_damper_energy",0,1,&x) == 1)
      fprintf(fp," spring and damper energy.......%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"joint_internal_energy",0,1,&x) == 1)
      fprintf(fp," joint internal energy..........%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"hourglass_energy",0,1,&x) == 1)
      fprintf(fp," hourglass energy ..............%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"system_damping_energy",0,1,&x) != 1) break;
    fprintf(fp," system damping energy..........%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"sliding_interface_energy",0,1,&x) != 1) break;
    fprintf(fp," sliding interface energy.......%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"external_work",0,1,&x) != 1) break;
    fprintf(fp," external work..................%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"eroded_kinetic_energy",0,1,&x) == 1)
      fprintf(fp," eroded kinetic energy..........%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"eroded_internal_energy",0,1,&x) == 1)
      fprintf(fp," eroded internal energy.........%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"eroded_hourglass_energy",0,1,&x) == 1)
      fprintf(fp," eroded hourglass energy........%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"total_energy",0,1,&x) != 1) break;
    fprintf(fp," total energy...................%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"energy_ratio",0,1,&x) != 1) break;
    fprintf(fp," total energy / initial energy..%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"energy_ratio_wo_eroded",0,1,&x) == 1)
      fprintf(fp," energy ratio w/o eroded energy.%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"global_x_velocity",0,1,&x) != 1) break;
    fprintf(fp," global x velocity..............%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"global_y_velocity",0,1,&x) != 1) break;
    fprintf(fp," global y velocity..............%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"global_z_velocity",0,1,&x) != 1) break;
    fprintf(fp," global z velocity..............%14.5E\n",x);
    if(lsda_read(handle,LSDA_INT,"number_of_nodes",0,1,&i) == 1) {
      fprintf(fp," number of nodes................%14d\n",i);
    }
    if(lsda_read(handle,LSDA_INT,"number_of_elements",0,1,&i) == 1) {
      fprintf(fp," number of elements.............%14d\n",i);
    }
    if(lsda_read(handle,LSDA_INT,"nzc",0,1,&i) != 1) break;
    fprintf(fp," time per zone cycle.(nanosec)..%14d\n",i);
    if(lsda_read(handle,LSDA_INT,"num_bad_shells",0,1,&i) == 1) {
      fprintf(fp,"\n\n number of shell elements that  \n");
      fprintf(fp," reached the minimum time step..%5d\n",i);
    }
    if(lsda_read(handle,LSDA_FLOAT,"added_mass",0,1,&x) == 1)
      fprintf(fp,"\n\n added mass.....................%14.5E\n",x);
    if(lsda_read(handle,LSDA_FLOAT,"percent_increase",0,1,&x) == 1)
      fprintf(fp," percentage increase............%14.5E\n",x);
/*
  Check for optional output of system RB properties
*/
    if(lsda_read(handle,LSDA_FLOAT,"total_mass",0,1,&x) == 1) {
      float t[9];
      fprintf(fp,"\n  m a s s   p r o p e r t i e s   o f   b o d y\n");
      fprintf(fp,"     total mass of body          =%15.8E\n",x);
      if(lsda_read(handle,LSDA_FLOAT,"x_mass_center",0,1,&x) != 1) break;
      fprintf(fp,"     x-coordinate of mass center =%15.8E\n",x);
      if(lsda_read(handle,LSDA_FLOAT,"y_mass_center",0,1,&x) != 1) break;
      fprintf(fp,"     y-coordinate of mass center =%15.8E\n",x);
      if(lsda_read(handle,LSDA_FLOAT,"z_mass_center",0,1,&x) != 1) break;
      fprintf(fp,"     z-coordinate of mass center =%15.8E\n\n",x);
      if(lsda_read(handle,LSDA_FLOAT,"inertia_tensor",0,9,&t) != 9) break;
      fprintf(fp,"     inertia tensor of body\n");
      fprintf(fp,"     row1=%15.4E%15.4E%15.4E\n",t[0],t[1],t[2]);
      fprintf(fp,"     row2=%15.4E%15.4E%15.4E\n",t[3],t[4],t[5]);
      fprintf(fp,"     row3=%15.4E%15.4E%15.4E\n\n\n",t[6],t[7],t[8]);
      if(lsda_read(handle,LSDA_FLOAT,"principal_inertias",0,3,&t) != 3) break;
      fprintf(fp,"\n     principal inertias of body\n");
      fprintf(fp,"     i11 =%15.4E\n",t[0]);
      fprintf(fp,"     i22 =%15.4E\n",t[1]);
      fprintf(fp,"     i33 =%15.4E\n\n",t[2]);
      if(lsda_read(handle,LSDA_FLOAT,"principal_directions",0,9,&t) != 9) break;
      fprintf(fp,"     principal directions\n");
      fprintf(fp,"     row1=%15.4E%15.4E%15.4E\n",t[0],t[1],t[2]);
      fprintf(fp,"     row2=%15.4E%15.4E%15.4E\n",t[3],t[4],t[5]);
      fprintf(fp,"     row3=%15.4E%15.4E%15.4E\n",t[6],t[7],t[8]);
      fprintf(fp,"\n ************************************************************\n\n");
    }
  }
  fclose(fp);
  if(nsw > 0) free(swe);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  SSSTAT file
*/
int translate_ssstat(int handle)
{
  int i,j,k,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256],dname[32];
  char types[20][32];
  char elnam[10];
  int nintcy,ityptc,ielprt,ipartc;
  int *systems;
  int *ids;
  int cycle,sub_system;
  float keg, ieg, heg, time, time_step;
  float *ke, *ie, *he, *xm, *ym, *zm, *ker, *ier;
  float *system_inertia;
  FILE *fp;
  LSDADir *dp = NULL;

  lsda_cd(handle,"/glstat/metadata");
  printf("Extracting SSSTAT data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"element_types",&typid,&length,&filenum);
  lsda_read(handle,LSDA_I1,"element_types",0,length,dirname);
/*
  parse dirname to split out the element types
*/
  for(i=j=k=0; i<length; i++) {
    if(dirname[i] == ',') {
      types[j][k]=0;
      j++;
      k=0;
    } else {
      types[j][k++]=dirname[i];
    }
  }
  types[j][k]=0;
/*
  Now data from ssstat
*/
  lsda_cd(handle,"/ssstat/metadata");
  lsda_queryvar(handle,"systems",&typid,&length,&filenum);
  num = length;

  systems = (int *) malloc(num*sizeof(int));
  ke = (float *) malloc(num*sizeof(float));
  ie = (float *) malloc(num*sizeof(float));
  he = (float *) malloc(num*sizeof(float));
  xm = (float *) malloc(num*sizeof(float));
  ym = (float *) malloc(num*sizeof(float));
  zm = (float *) malloc(num*sizeof(float));
  ker = (float *) malloc(num*sizeof(float));
  ier = (float *) malloc(num*sizeof(float));

  system_inertia = (float *) malloc(25*num*sizeof(float));

  lsda_read(handle,LSDA_INT,"systems",0,num,systems);
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  ids = (int *) malloc(length*sizeof(int));
  lsda_read(handle,LSDA_INT,"ids",0,length,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%sssstat",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/ssstat/metadata",fp);
/*
  susbsystem info
*/
  for(i=j=0; i<num; i++) {
    fprintf(fp,"\n\n subsystem definition ID=%12d part ID list:\n",i+1);
    for(k=0; k<systems[i]; k++) {
      fprintf(fp,"%10d",ids[j++]);
      if((k+1)%8 == 0) fprintf(fp,"\n");
    }
    if(k%8 != 0) fprintf(fp,"\n");
  }
  free(ids);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/ssstat",dp,dname)) != NULL; state++) {
/*
  First get the subsystem data
*/
    sub_system=num;
    if(lsda_read(handle,LSDA_INT,"ncycle",0,1,&nintcy) != 1) break;
    if(lsda_read(handle,LSDA_INT,"elmtyp",0,1,&ityptc) != 1) break;
    if(lsda_read(handle,LSDA_INT,"elemid",0,1,&ielprt) != 1) break;
    if(lsda_read(handle,LSDA_INT,"partid",0,1,&ipartc) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"time_step",0,1,&time_step) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"kin_energy_g",0,1,&keg) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"int_energy_g",0,1,&ieg) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"hgl_energy_g",0,1,&heg) != 1) heg = 0.;

    if(lsda_read(handle,LSDA_FLOAT,"kinetic_energy",0,num,ke) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,num,ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hourglass_energy",0,num,he) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_momentum",0,num,xm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_momentum",0,num,ym) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_momentum",0,num,zm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"kinetic_energy_ratios",0,num,ker) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"internal_energy_ratios",0,num,ier) != num) break;

    if(lsda_read(handle,LSDA_FLOAT,"subsystem_inertia_info",0,25*num,system_inertia) != 25*num) sub_system=0;
/*
  Now get the necessary global data from the glstat directory.
*/

    strcpy(elnam,"          ");
    if(ityptc == 2) strcpy(elnam,"solid");
    if(ityptc == 3) strcpy(elnam,"beam");
    if(ityptc == 4) strcpy(elnam,"shell");
    if(ityptc == 5) strcpy(elnam,"thk shell");
    fprintf(fp,"\n\n dt of cycle %8d is controlled by %s %10d of part %10d\n",nintcy,&elnam,ielprt,ipartc);
    fprintf(fp,"\n time........................   %14.5E\n",time);
    fprintf(fp," time step...................   %14.5E\n",time_step);
    fprintf(fp," kinetic energy global.......   %14.5E\n",keg);
    fprintf(fp," kinetic energy subsystems ..");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,ke[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");
    fprintf(fp," internal energy global......   %14.5E\n",ieg);
    fprintf(fp," internal energy subsystems .");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,ie[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");
    fprintf(fp," hourglass energy global ....   %14.5E\n",heg);
    fprintf(fp," hourglass energy subsystems ");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,he[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");
    fprintf(fp," kinetic energy ratios ......");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,ker[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");
    fprintf(fp," internal energy ratios .....");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,ier[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");
    fprintf(fp," x-momentum subsystems ......");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,xm[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");
    fprintf(fp," y-momentum subsystems ......");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,ym[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");
    fprintf(fp," z-momentum subsystems ......");
    for(i=0; i<num; i++) {
      if(i>0 && i%3 == 0) fprintf(fp,"                             ");
      fprintf(fp,"%3d.%13.5E",i+1,zm[i]);
      if((i+1)%3 == 0) fprintf(fp,"\n");
    }
    if(num%3 != 0) fprintf(fp,"\n");

    for(i=0; i<sub_system; i++) {
      k = 25*i;
      fprintf(fp,"\n\n subsystem:%5d\n\n",i+1);
      fprintf(fp,"     total mass of subsystem     =%15.8E\n",system_inertia[k  ]);
      fprintf(fp,"     x-coordinate of mass center =%15.8E\n",system_inertia[k+1]);
      fprintf(fp,"     y-coordinate of mass center =%15.8E\n",system_inertia[k+2]);
      fprintf(fp,"     z-coordinate of mass center =%15.8E\n",system_inertia[k+3]);
      fprintf(fp,"\n     inertia tensor in global coordinates\n");
      fprintf(fp,"     row1=%15.4E%15.4E%15.4E\n",system_inertia[k+ 4],system_inertia[k+ 5],system_inertia[k+ 6]);
      fprintf(fp,"     row2=%15.4E%15.4E%15.4E\n",system_inertia[k+ 7],system_inertia[k+ 8],system_inertia[k+ 9]);
      fprintf(fp,"     row3=%15.4E%15.4E%15.4E\n",system_inertia[k+10],system_inertia[k+11],system_inertia[k+12]);
      fprintf(fp,"\n     principal inertias\n");
      fprintf(fp,"     i11 =%14.4E\n",system_inertia[k+13]);
      fprintf(fp,"     i22 =%14.4E\n",system_inertia[k+14]);
      fprintf(fp,"     i33 =%14.4E\n",system_inertia[k+15]);
      fprintf(fp,"\n     principal directions\n");
      fprintf(fp,"     row1=%15.4E%15.4E%15.4E\n",system_inertia[k+16],system_inertia[k+17],system_inertia[k+18]);
      fprintf(fp,"     row2=%15.4E%15.4E%15.4E\n",system_inertia[k+19],system_inertia[k+20],system_inertia[k+21]);
      fprintf(fp,"     row3=%15.4E%15.4E%15.4E\n",system_inertia[k+22],system_inertia[k+23],system_inertia[k+24]);
    }
  }
  fclose(fp);
  free(system_inertia);
  free(ier);
  free(ker);
  free(zm);
  free(ym);
  free(xm);
  free(he);
  free(ie);
  free(ke);
  free(systems);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  DEFORC file
*/
int translate_deforc(int handle)
{
  int i,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int *ids, *irot;
  float *fx,*fy,*fz,*rf,*disp;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/deforc/metadata") == -1) return 0;
  printf("Extracting DEFORC data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;

  ids  = (int *) malloc(num*sizeof(int));
  irot = (int *) malloc(num*sizeof(int));
  fx  = (float *) malloc(num*sizeof(float));
  fy  = (float *) malloc(num*sizeof(float));
  fz  = (float *) malloc(num*sizeof(float));
  rf  = (float *) malloc(num*sizeof(float));
  disp = (float *) malloc(num*sizeof(float));
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
  lsda_read(handle,LSDA_INT,"irot",0,num,irot);
/*
  open file and write header
*/
  sprintf(output_file,"%sdeforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/deforc/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/deforc",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_force",0,num,fx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_force",0,num,fy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_force",0,num,fz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"resultant_force",0,num,rf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"displacement",0,num,disp) != num) break;

    fprintf(fp,"\n time.........................%14.5E\n",time);
    for(i=0; i<num; i++) {
      fprintf(fp," spring/damper number.........%10d\n",ids[i]);
      if(irot[i] == 0) {
        fprintf(fp," x-force......................%14.5E\n",fx[i]);
        fprintf(fp," y-force......................%14.5E\n",fy[i]);
        fprintf(fp," z-force......................%14.5E\n",fz[i]);
        fprintf(fp," resultant force..............%14.5E\n",rf[i]);
        fprintf(fp," change in length.............%14.5E\n",disp[i]);
      } else {
        fprintf(fp," x-moment.....................%14.5E\n",fx[i]);
        fprintf(fp," y-moment.....................%14.5E\n",fy[i]);
        fprintf(fp," z-moment.....................%14.5E\n",fz[i]);
        fprintf(fp," resultant moment.............%14.5E\n",rf[i]);
        fprintf(fp," relative rotation............%14.5E\n",disp[i]);
      }
    }
  }
  fclose(fp);
  free(disp);
  free(rf);
  free(fz);
  free(fy);
  free(fx);
  free(irot);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  MATSUM file
*/
int translate_matsum(int handle)
{
  int i,typid,num,filenum,state;
  int have_mass, have_he, have_eroded;
  LSDA_Length length;
  char dirname[256];
  int *ids;
  float *ke, *ie, *he, *mass, *xm, *ym, *zm, *xrbv, *yrbv, *zrbv;
  float *eke, *eie;
  float mm,time,x;
  int mid;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/matsum/metadata") == -1) return 0;
  printf("Extracting MATSUM data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;

  ids = (int *) malloc(num*sizeof(int));
  ie = (float *) malloc(num*sizeof(float));
  ke = (float *) malloc(num*sizeof(float));
  eie = (float *) malloc(num*sizeof(float));
  eke = (float *) malloc(num*sizeof(float));
  mass = (float *) malloc(num*sizeof(float));
  xm = (float *) malloc(num*sizeof(float));
  ym = (float *) malloc(num*sizeof(float));
  zm = (float *) malloc(num*sizeof(float));
  xrbv = (float *) malloc(num*sizeof(float));
  yrbv = (float *) malloc(num*sizeof(float));
  zrbv = (float *) malloc(num*sizeof(float));
  he = (float *) malloc(num*sizeof(float));
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%smatsum",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/matsum/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/matsum",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,num,ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"kinetic_energy",0,num,ke) != num) break;

    lsda_queryvar(handle,"eroded_internal_energy",&have_eroded,&length,&filenum);
    if(have_eroded > 0) {
      if(lsda_read(handle,LSDA_FLOAT,"eroded_internal_energy",0,num,eie) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"eroded_kinetic_energy",0,num,eke) != num) break;
    }
    lsda_queryvar(handle,"mass",&have_mass,&length,&filenum);
    if(have_mass > 0)
      if(lsda_read(handle,LSDA_FLOAT,"mass",0,num,mass) != num) break;
    lsda_queryvar(handle,"hourglass_energy",&have_he,&length,&filenum);
    if(have_he > 0)
      if(lsda_read(handle,LSDA_FLOAT,"hourglass_energy",0,num,he) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_momentum",0,num,xm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_momentum",0,num,ym) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_momentum",0,num,zm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_rbvelocity",0,num,xrbv) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_rbvelocity",0,num,yrbv) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_rbvelocity",0,num,zrbv) != num) break;

    fprintf(fp,"\n\n time =%13.4E\n",time);
/*
  This is an...."interesting" bit of a hack.  The kinetic energies of lumped
  masses can optionally be output, in which case they are supposed to show
  up in the ascii matsum file as material 0....ugh.
*/
    if(lsda_read(handle,LSDA_FLOAT,"lumped_kinetic_energy",0,1,&x) == 1) {
      fprintf(fp," mat.#=%5d   ",0);
      fprintf(fp,"          inten=%13.4E     kinen=%13.4E",0.0,x);
      fprintf(fp,"     eroded_ie=%13.4E     eroded_ke=%13.4E\n",0.0,0.0);
    }
    for(i=0; i<num; i++) {
      if(ids[i] < 100000 )
        fprintf(fp," mat.#=%5d   ",ids[i]);
      else
        fprintf(fp," mat.#=%8d",ids[i]);
      fprintf(fp,"          inten=%13.4E     kinen=%13.4E",ie[i],ke[i]);
      if(have_eroded>0)
        fprintf(fp,"     eroded_ie=%13.4E     eroded_ke=%13.4E",eie[i],eke[i]);
      fprintf(fp,"\n");
      fprintf(fp," x-mom=%13.4E     y-mom=%13.4E     z-mom=%13.4E\n",
         xm[i],ym[i],zm[i]);
      fprintf(fp," x-rbv=%13.4E     y-rbv=%13.4E     z-rbv=%13.4E\n",
         xrbv[i],yrbv[i],zrbv[i]);
      if(have_he>0) {
        fprintf(fp,"                         hgeng=%13.4E     ",he[i]);
        if(have_mass>0)
          fprintf(fp,"+mass=%13.4E\n",mass[i]);
        else
          fprintf(fp,"\n");

        fprintf(fp,"\n\n");
      } else if(have_mass>0) {
          fprintf(fp,"                                                 ");
          fprintf(fp,"+mass=%13.4E\n",mass[i]);
      }
    }
    lsda_queryvar(handle,"brick_id",&typid,&length,&filenum);
    if(typid > 0) {
      lsda_read(handle,LSDA_FLOAT,"max_brick_mass",0,1,&mm);
      lsda_read(handle,LSDA_INT,"brick_id",0,1,&mid);
      fprintf(fp,"\n\n       Maximum mass increase in brick elements      =");
      fprintf(fp,"%13.4E ID=%8d\n",mm,mid);
    }
    lsda_queryvar(handle,"beam_id",&typid,&length,&filenum);
    if(typid > 0) {
      lsda_read(handle,LSDA_FLOAT,"max_beam_mass",0,1,&mm);
      lsda_read(handle,LSDA_INT,"beam_id",0,1,&mid);
      fprintf(fp,"\n\n       Maximum mass increase in beam elements       =");
      fprintf(fp,"%13.4E ID=%8d\n",mm,mid);
    }
    lsda_queryvar(handle,"shell_id",&typid,&length,&filenum);
    if(typid > 0) {
      lsda_read(handle,LSDA_FLOAT,"max_shell_mass",0,1,&mm);
      lsda_read(handle,LSDA_INT,"shell_id",0,1,&mid);
      fprintf(fp,"\n\n       Maximum mass increase in shell elements      =");
      fprintf(fp,"%13.4E ID=%8d\n",mm,mid);
    }
    lsda_queryvar(handle,"thick_shell_id",&typid,&length,&filenum);
    if(typid > 0) {
      lsda_read(handle,LSDA_FLOAT,"max_thick_shell_mass",0,1,&mm);
      lsda_read(handle,LSDA_INT,"thick_shell_id",0,1,&mid);
      fprintf(fp,"\n\n       Maximum mass increase in thick shell elements=");
      fprintf(fp,"%13.4E ID=%8d\n",mm,mid);
    }
  }
  fclose(fp);
  free(he);
  free(zrbv);
  free(yrbv);
  free(xrbv);
  free(zm);
  free(ym);
  free(xm);
  free(mass);
  free(eke);
  free(eie);
  free(ke);
  free(ie);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  NCFORC file
*/
typedef struct {
  int master;      /* 0 for slave, 1 for master */
  int inum;        /* interface number for output */
  int *ids;        /* user ids for nodes */
  int n;           /* number of nodes */
  char legend[80];
  char dname[32];  /* so I don't have to keep rebuilding it...*/
} NCFDATA;

int comp_ncn(const void *v1, const void *v2)
{
  NCFDATA *p1 = (NCFDATA *) v1;
  NCFDATA *p2 = (NCFDATA *) v2;
  if(p1->inum != p2->inum) return (p1->inum - p2->inum);
  return (p1->master - p2->master);
}

int translate_ncforc(int handle)
{
/*
  This one is a bit strange in that there are separate dirs
  for each slave/master side of each interface.  The only
  way to tell how many there are is to count the dirs.
  Fortunately, they appear in the same order we want to
  output them in.....
  NOT!  readdir now returns things in alphabetic order, so we will
  have to resort them according to their number.....
*/
  NCFDATA *dp;
  int ndirs;
  LSDADir *ldp;
  int i,j,k,bpt,typid,filenum,state,nmax;
  LSDA_Length length;
  char dirname[256];
  float *xf,*yf,*zf,*p, *x, *y, *z;
  float time;
  FILE *fp;
  int have_legend=0, lterms=0;

  if(!(ldp = lsda_opendir(handle,"/ncforc"))) return 0;
  printf("Extracting NCFORC data\n");
  for(ndirs=0; ;ndirs++) {
    lsda_readdir(ldp,dirname,&typid,&length,&filenum);
    if(dirname[0]==0)  break ;   /* end of listing */
  }
  lsda_closedir(ldp);

  dp = (NCFDATA *) malloc(ndirs * sizeof(NCFDATA));

  ldp = lsda_opendir(handle,"/ncforc");
  for(i=0;i<ndirs ;i++) {
    lsda_readdir(ldp,dirname,&typid,&length,&filenum);
    if(strcmp(dirname,"metadata")==0) {  /* skip this one */
      i--;
      continue;
    }
    dp[i].master = (dirname[0] == 'm');
    if(dp[i].master)
      sscanf(dirname+7,"%d",&dp[i].inum);
    else
      sscanf(dirname+6,"%d",&dp[i].inum);
    sprintf(dp[i].dname,"/ncforc/%s",dirname);
  }
  lsda_closedir(ldp);
  qsort(dp,ndirs,sizeof(NCFDATA),comp_ncn);
/*
  Ok, now go through each directory and get the list of user ids.
  Also, build up legend info if we have any
*/
  for(i=nmax=0; i<ndirs; i++) {
    sprintf(dirname,"%s/metadata",dp[i].dname);
    lsda_cd(handle,dirname);
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    if(typid > 0) {
    dp[i].n = length;
    if(nmax < length) nmax = length;
    dp[i].ids = (int *) malloc(dp[i].n*sizeof(int));
      lsda_read(handle,LSDA_INT,"ids",0,length,dp[i].ids);
    } else {
      dp[i].n = 0;
      dp[i].ids = NULL;
    }
    lsda_queryvar(handle,"legend",&typid,&length,&filenum);
    if(typid > 0) {
      k=length;
      bpt=sizeof(dp[i].legend);
      if(k > bpt) k=bpt;
      lsda_read(handle,LSDA_I1,"legend",0,k,dp[i].legend);
      for( ; k < bpt; k++)
        dp[i].legend[k]=' ';
      have_legend=1;
    } else
      dp[i].legend[0]=0;
  }
  if(nmax == 0) return 0;  /* ???? */
  xf = (float *) malloc(nmax*sizeof(float));
  yf = (float *) malloc(nmax*sizeof(float));
  zf = (float *) malloc(nmax*sizeof(float));
  p  = (float *) malloc(nmax*sizeof(float));
  x  = (float *) malloc(nmax*sizeof(float));
  y  = (float *) malloc(nmax*sizeof(float));
  z  = (float *) malloc(nmax*sizeof(float));
/*
  open file and write header
*/
  sprintf(output_file,"%sncforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,dirname,fp);
  if(have_legend) {
    fprintf(fp,"\n{BEGIN LEGEND\n");
    fprintf(fp," Entity #        Title\n");
    for(i=j=0; i<ndirs; i++) {
      if(dp[i].legend[0] && dp[i].inum != j) {
        fprintf(fp,"%9d     %.80s\n",dp[i].inum,dp[i].legend);
        j=dp[i].inum;
      }
    }
    fprintf(fp,"{END LEGEND}\n\n");
  }
/*
  Loop through time states and write each one.
*/
  for(state=1; ; state++) {
    for(k=0; k<ndirs; k++) {
      if(state<=999999)
        sprintf(dirname,"%s/d%6.6d",dp[k].dname,state);
      else
        sprintf(dirname,"%s/d%8.8d",dp[k].dname,state);

      lsda_queryvar(handle,dirname,&typid,&length,&filenum);
      if(typid != 0) goto done;
      lsda_cd(handle,dirname);
      if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) goto done;
/*
   If this is an eroding interface, the number of nodes can change
   each time.  We know if there is and "ids" array in this directory...
*/
      lsda_queryvar(handle,"ids",&typid,&length,&filenum);
      lterms = (int) length;
      if(typid > 0 && lterms > 0) {
        dp[k].n = lterms;
        if(dp[k].ids) free(dp[k].ids);
        dp[k].ids = (int *) malloc(dp[k].n*sizeof(int));
        lsda_read(handle,LSDA_INT,"ids",0,lterms,dp[k].ids);
        if(lterms > nmax) {
          nmax = lterms;
          free(xf);
          free(yf);
          free(zf);
          free(p);
          free(x);
          free(y);
          free(z);
          xf = (float *) malloc(nmax*sizeof(float));
          yf = (float *) malloc(nmax*sizeof(float));
          zf = (float *) malloc(nmax*sizeof(float));
          p  = (float *) malloc(nmax*sizeof(float));
          x  = (float *) malloc(nmax*sizeof(float));
          y  = (float *) malloc(nmax*sizeof(float));
          z  = (float *) malloc(nmax*sizeof(float));
        }
      }
      if(dp[k].n > 0) {
      if(lsda_read(handle,LSDA_FLOAT,"x_force",0,dp[k].n,xf) != dp[k].n) goto done;
      if(lsda_read(handle,LSDA_FLOAT,"y_force",0,dp[k].n,yf) != dp[k].n) goto done;
      if(lsda_read(handle,LSDA_FLOAT,"z_force",0,dp[k].n,zf) != dp[k].n) goto done;
      if(lsda_read(handle,LSDA_FLOAT,"pressure",0,dp[k].n,p) != dp[k].n) goto done;
      if(lsda_read(handle,LSDA_FLOAT,"x",0,dp[k].n,x) != dp[k].n) goto done;
      if(lsda_read(handle,LSDA_FLOAT,"y",0,dp[k].n,y) != dp[k].n) goto done;
        if(lsda_read(handle,LSDA_FLOAT,"z",0,dp[k].n,z) != dp[k].n) goto done;
      }

      fprintf(fp,"\n\n\n forces (t=%11.5E) for interface%10d %s side\n\n",
        time,dp[k].inum,(dp[k].master ? "master" : "slave "));
      fprintf(fp,"     node           x-force/      y-force/      z-force/");
      fprintf(fp,"     pressure/\n");
      fprintf(fp,"                   coordinate    coordinate    coordinate\n");
      for(i=0; i<dp[k].n; i++) {
        fprintf(fp,"%10d      %14.5E%14.5E%14.5E%14.5E\n",dp[k].ids[i],
          xf[i],yf[i],zf[i],p[i]);
        fprintf(fp,"                %14.5E%14.5E%14.5E\n",x[i],y[i],z[i]);
      }
    }
  }
done:
  fclose(fp);
  free(z);
  free(y);
  free(x);
  free(p);
  free(zf);
  free(yf);
  free(xf);
  for(i=0; i<ndirs; i++)
    if(dp[i].ids) free(dp[i].ids);
  free(dp);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  RCFORC file
*/
int translate_rcforc(int handle)
{
  int i,j,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[32];
  int *ids;
  int *sides, *single, nsingle, nsout;
  int *tcount;
  float *xf,*yf,*zf,*mass;
  float *xm,*ym,*zm;
  float *tarea;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/rcforc/metadata") == -1) return 0;
  printf("Extracting RCFORC data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  lsda_queryvar(handle,"singlesided",&typid,&length,&filenum);
  if(typid > 0)
    nsingle = length/2;
  else
    nsingle = 0;

  ids = (int *) malloc(num*sizeof(int));
  sides = (int *) malloc(num*sizeof(int));
  xf = (float *) malloc(num*sizeof(float));
  yf = (float *) malloc(num*sizeof(float));
  zf = (float *) malloc(num*sizeof(float));
  mass = (float *) malloc(num*sizeof(float));
  lsda_queryvar(handle,"/rcforc/d000001/x_moment",&typid,&length,&filenum);
  if(length > 0) {
    xm = (float *) malloc(num*sizeof(float));
    ym = (float *) malloc(num*sizeof(float));
    zm = (float *) malloc(num*sizeof(float));
  } else {
    xm = ym = zm = NULL;
  }
  lsda_queryvar(handle,"/rcforc/d000001/tie_area",&typid,&length,&filenum);
  if(length > 0) {
    tcount = (int *) malloc(num*sizeof(int));
    tarea  = (float *) malloc(num*sizeof(float));
  } else {
    tcount = NULL;
    tarea  = NULL;
  }
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
  lsda_read(handle,LSDA_INT,"side",0,num,sides);

/* in case of some illegal contact definition, ie no slave/master in the
   input deck which contact ID will be 0 */
  lsda_queryvar(handle,"/rcforc/d000001/x_force",&typid,&length,&filenum);
  if(length < num) {
    j=0;
    for(i=0; i<num; i++) { 
      if(ids[i] != 0 ) { 
        ids[j]  =ids[i];
        sides[j]=sides[i];
        j++;
      } else {
        printf (" *** Warning: Please check contact definition %d\n",i/2);
    } }
    num = j;
  }

  if(nsingle) {
    single = (int *) malloc(nsingle*2*sizeof(int));
    lsda_read(handle,LSDA_INT,"singlesided",0,2*nsingle,single);
  }

/*
  open file and write header
*/
  sprintf(output_file,"%srcforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/rcforc/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/rcforc",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_force",0,num,xf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_force",0,num,yf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_force",0,num,zf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"mass",0,num,mass) != num) break;
    if(xm) {
      if(lsda_read(handle,LSDA_FLOAT,"x_moment",0,num,xm) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"y_moment",0,num,ym) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"z_moment",0,num,zm) != num) break;
    }
    if(tcount) {
      if(lsda_read(handle,LSDA_FLOAT,"tie_area",0,num,tarea) != num) break;
      if(lsda_read(handle,LSDA_INT,"tie_count",0,num,tcount) != num) break;
    }

    nsout=0;
    for(i=0; i<num; i++) {
      while(nsout < nsingle && i == single[2*nsout]) {
        fprintf(fp," interface number, %9d,is single surface-resultants are undefined\n",single[2*nsout+1]);
        nsout++;
      }
      if(sides[i])
        fprintf(fp,"  master%11d time",ids[i]);
      else
        fprintf(fp,"  slave %11d time",ids[i]);
      fprintf(fp,"%12.5E  x %12.5E  y %12.5E  z %12.5E mass %12.5E",
        time,xf[i],yf[i],zf[i],mass[i]);
      if(xm) {
        fprintf(fp,"  mx %12.5E  my %12.5E  mz %12.5E",xm[i],ym[i],zm[i]);
      }
      if(tcount && tcount[i] >= 0) {
        if(!xm)
          fprintf(fp,"                                                   ");
        fprintf(fp,"  tied %10d  tied area %12.5E",tcount[i],tarea[i]);
      }
      fprintf(fp,"\n");
    }
    while(nsout < nsingle && num == single[2*nsout]) {
      fprintf(fp,"  interface number, %11d,is single surface-resultants are undefined\n",single[2*nsout+1]);
      nsout++;
    }
  }
  fclose(fp);
  free(mass);
  free(zf);
  free(yf);
  free(xf);
  free(sides);
  if(nsingle) free(single);
  if(xm) {
     free(xm);
     free(ym);
     free(zm);
  }
  if(tcount) {
    free(tcount);
    free(tarea);
  }
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  SPCFORC file
*/
int translate_spcforc(int handle)
{
  int i,j,typid,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int *tids,*rids,*id,*mid,numt,numr;
  float *xf,*yf,*zf,*xm,*ym,*zm,xtot,ytot,ztot;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/spcforc/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_queryvar(handle,"force_ids",&typid,&length,&filenum);
  if(typid > 0) {
    numt = length;
    tids = (int *) malloc(numt*sizeof(int));
    id   = (int *) malloc(numt*sizeof(int));
    xf   = (float *) malloc(numt*sizeof(float));
    yf   = (float *) malloc(numt*sizeof(float));
    zf   = (float *) malloc(numt*sizeof(float));
    lsda_read(handle,LSDA_INT,"force_ids",0,length,tids);
  } else {
    numt = 0;
  }
  lsda_queryvar(handle,"moment_ids",&typid,&length,&filenum);
  if(typid > 0) {
    numr = length;
    rids = (int *) malloc(numr*sizeof(int));
    mid  = (int *) malloc(numr*sizeof(int));
    xm   = (float *) malloc(numr*sizeof(float));
    ym   = (float *) malloc(numr*sizeof(float));
    zm   = (float *) malloc(numr*sizeof(float));
    lsda_read(handle,LSDA_INT,"moment_ids",0,length,rids);
  } else {
    numr = 0;
  }

  if(numt == 0 && numr == 0) return 0;

  lsda_queryvar(handle,"spc_ids",&typid,&length,&filenum);
  if(typid > 0) {
    lsda_read(handle,LSDA_INT,"spc_ids",0,length,id);
  } else {
    memset(id,0,numt*sizeof(int));
  }
  lsda_queryvar(handle,"spc_mids",&typid,&length,&filenum);
  if(typid > 0) {
    lsda_read(handle,LSDA_INT,"spc_mids",0,length,mid);
  } else {
	  if(numr)
		memset(mid,0,numr*sizeof(int));
  }
/*
  open file and write header
*/
  printf("Extracting SPCFORC data\n");
  sprintf(output_file,"%sspcforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/spcforc/metadata",fp);
/*
  This may have worked for the one test problem that some user submitted,
  but it is seg faulting when the id[] array isn't just "1, 2, 3,..." and
  it doesn't account at all for rotational SPCs, so it can't be completely
  correct.  I don't know what "problem" it was trying to solve, so just
  remove it....
  output_legend_nosort(handle,fp,1,1,id);
*/
  output_legend(handle,fp,1,1);
  fprintf(fp," single point constraint forces\n\n");
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/spcforc",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",        0,  1,&time) != 1) break;
    if(numt) {
      if(lsda_read(handle,LSDA_FLOAT,"x_force",0,numt,xf) != numt) break;
      if(lsda_read(handle,LSDA_FLOAT,"y_force",0,numt,yf) != numt) break;
      if(lsda_read(handle,LSDA_FLOAT,"z_force",0,numt,zf) != numt) break;
    }
    if(numr) {
      if(lsda_read(handle,LSDA_FLOAT,"x_moment",0,numr,xm) != numr) break;
      if(lsda_read(handle,LSDA_FLOAT,"y_moment",0,numr,ym) != numr) break;
      if(lsda_read(handle,LSDA_FLOAT,"z_moment",0,numr,zm) != numr) break;
    }
    if(lsda_read(handle,LSDA_FLOAT,"x_resultant",0,1,&xtot) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_resultant",0,1,&ytot) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_resultant",0,1,&ztot) != 1) break;
/*
  Now, it appears that the serial code normally outputs them in increasing
  node order, with translations before moments if both appear.  So we will
  put them out that way also for compatibility
*/
    fprintf(fp," output at time =%14.5E\n",time);
    i=j=0;
    while(i<numt || j<numr) {
      if(i < numt) {
        if(j >= numr || tids[i] <= rids[j]) {
          fprintf(fp," node=%8d local x,y,z forces =%14.4E%14.4E%14.4E  setid=%8d\n",
           tids[i],xf[i],yf[i],zf[i],id[i]);
          i++;
        }
      }
      if(j < numr) {
        if(i >= numt || tids[i] > rids[j]) {
          fprintf(fp," node=%8d local x,y,z moments=%14.4E%14.4E%14.4E  setid=%8d\n",
           rids[j],xm[j],ym[j],zm[j],mid[j]);
          j++;
        }
      }
    }
    fprintf(fp,"             force resultants   =%14.4E%14.4E%14.4E\n",xtot,ytot,ztot);
  }
  fclose(fp);
  if(numr) {
    free(zm);
    free(ym);
    free(xm);
    free(mid);
    free(rids);
  }
  if(numt) {
    free(zf);
    free(yf);
    free(xf);
    free(id);
    free(tids);
  }
  printf("      %d states extracted\n",state-1);
  return 0;
}
/* convert floating point value to a major hack of a string format
 * to match the strange format used by the ascii file */
char *tochar(float value,int len)
{
  static char s[20];
  int i,to,from;

  sprintf(s,"%12.5E",value);
  from = 9;
  to = len-2;
  for(i=0; i<4; i++)
    s[to+i]=s[from+i];
  return s+1;
}
/*
  SWFORC file
*/
int translate_swforc(int handle)
{
  int i,j,typid,num,filenum,state;
  LSDA_Length length,length2;
  char dirname[256];
  int *ids,*type,nnnodal;
  float *axial,*shear,*ftime,*emom,*swlen,*fflag;
  float *rmom,*torsion,*fp1,*fn1,*fp2,*fn2;
  float *fmax,*tfmax;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/swforc/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  ids = (int *) malloc(num*sizeof(int));
  type = (int *) malloc(num*sizeof(int));
  fflag = (float *) malloc(num*sizeof(float));
  axial = (float *) malloc(num*sizeof(float));
  shear = (float *) malloc(num*sizeof(float));
  i=sizeof(float) > sizeof(int) ? sizeof(float) : sizeof(int);
  ftime = (float *) malloc(num*i);
  swlen = (float *) malloc(num*sizeof(float));
  lsda_read(handle,LSDA_INT,"ids",0,length,ids);
  lsda_read(handle,LSDA_INT,"types",0,length,type);
  lsda_cd(handle,"/swforc/d000001");
  lsda_queryvar(handle,"emom",&typid,&length2,&filenum);
  if(typid > 0) {
    nnnodal = length2;
    emom = (float *) malloc(nnnodal*sizeof(float));
  } else {
    nnnodal = 0;
  }
  lsda_queryvar(handle,"resultant_moment",&typid,&length2,&filenum);
  if(typid > 0) {
    rmom = (float *) malloc(num*sizeof(float));
  } else {
    rmom = NULL;
  }
  lsda_queryvar(handle,"torsion",&typid,&length2,&filenum);
  if(typid > 0) {
    torsion = (float *) malloc(num*sizeof(float));
  } else {
    torsion = NULL;
  }
  lsda_queryvar(handle,"fp1",&typid,&length2,&filenum);
  if(typid > 0) {
    fp1 = (float *) malloc(num*sizeof(float));
    fn1 = (float *) malloc(num*sizeof(float));
    fp2 = (float *) malloc(num*sizeof(float));
    fn2 = (float *) malloc(num*sizeof(float));
  } else {
    fp1 = NULL;
    fn1 = NULL;
    fp2 = NULL;
    fn2 = NULL;
  }
  lsda_queryvar(handle,"max_failure",&typid,&length2,&filenum);
  if(typid > 0) {
    fmax = (float *) malloc(num*sizeof(float));
    tfmax = (float *) malloc(num*sizeof(float));
  } else {
    fmax = NULL;
    tfmax = NULL;
  }
/*
  open file and write header
*/
  printf("Extracting SWFORC data\n");
  sprintf(output_file,"%sswforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/swforc/metadata",fp);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/swforc",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",        0,  1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"axial",       0,num,axial) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"shear",       0,num,shear) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"failure",0,num,fflag) != num) {
    /* check for older file, which had integer flags stored as "failure_flag" */
      int *iflag = (int *) ftime;
      int i;
      if(lsda_read(handle,LSDA_INT,"failure_flag",0,num,iflag) != num) break;
    /* convert integers (0/1) to floats */
      for(i=0; i<num; i++)
        fflag[i] = (float) iflag[i];
    }
    if(lsda_read(handle,LSDA_FLOAT,"failure_time",0,num,ftime) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"length",0,num,swlen) != num) {
      memset(swlen,0,num*sizeof(float));
    }
    if(nnnodal)
      if(lsda_read(handle,LSDA_FLOAT,"emom",0,nnnodal,emom) != nnnodal) break;
    if(rmom!=NULL)
      if(lsda_read(handle,LSDA_FLOAT,"resultant_moment",0,num,rmom) != num) {
        free(rmom);
        rmom = NULL;
      }
    if(torsion!=NULL)
      if(lsda_read(handle,LSDA_FLOAT,"torsion",0,num,torsion) != num) {
      free(torsion);
      torsion = NULL;
    }
    if(fp1!=NULL) {
      if(lsda_read(handle,LSDA_FLOAT,"fp1",0,num,fp1) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"fn1",0,num,fn1) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"fp2",0,num,fp2) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"fn2",0,num,fn2) != num) break;
    }
    if(fmax != NULL) {
      if(lsda_read(handle,LSDA_FLOAT,"max_failure",0,num,fmax) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"max_failure_time",0,num,tfmax) != num) break;
    }
    fprintf(fp,"\n constraint #      axial        shear         time");
    fprintf(fp,"  failure                                       length");
    fprintf(fp,"  rslt moment      torsion");
    if (fp1!=NULL)
    {
    fprintf(fp,"          fp1          fn1          fp2          fn2");
    }
    fprintf(fp,"\n");
    for(i=j=0; i<num; i++) {
      if(type[i] == 3 || type[i] == 6)
      fprintf(fp,"      %5d%13.5E%13.5E%13.5E %8.4f    ",
       i+1,axial[i],shear[i],time,fflag[i]);
      else
      if(fflag[i] < 1.0)
          fprintf(fp,"      %5d%13.5E%13.5E%13.5E  %-7s    ",
           i+1,axial[i],shear[i],time,tochar2(fflag[i],7));
      else
        fprintf(fp,"      %5d%13.5E%13.5E%13.5E  failure    ",
         i+1,axial[i],shear[i],time);
      if(type[i] == 0) 
      {  
        fprintf(fp,"constraint/weld  ID %8d",ids[i]);
        if(swlen[i] > 0.0)
	  fprintf(fp,"%13.5E",swlen[i]);
        if(rmom != NULL)
	  fprintf(fp,"%13.5E",rmom[i]);
	fprintf(fp,"\n");
      }
      else if(type[i] == 1)
        fprintf(fp,"generalized weld ID %8d\n",ids[i]);
      else if(type[i] == 2) {
        fprintf(fp,"spotweld beam  ID   %8d",ids[i]);
        if(fmax != NULL && fmax[i] > 0.)
        {
          fprintf(fp,"%13.5E",swlen[i]);
          if(rmom != NULL)
            fprintf(fp,"%13.5E",rmom[i]);
          fprintf(fp," fcmx=%s",tochar2(fmax[i],7));
          fprintf(fp," t=%s\n",tochar2(tfmax[i],8));
        }
        else if(ftime[i] > 0.0)
        {
          fprintf(fp,"               failure time=%12.4E\n",ftime[i]);
        }
        else
        {
          fprintf(fp,"%13.5E",swlen[i]);
          if(rmom != NULL)
            fprintf(fp,"%13.5E",rmom[i]);
          else
            fprintf(fp,"             ");
          if(fp1 != NULL)
          {
            fprintf(fp,"             %13.5E%13.5E%13.5E%13.5E",fp1[i],fn1[i],fp2[i],fn2[i]);
          }
          fprintf(fp,"\n");
        }
      } else if(type[i] == 3) {
        fprintf(fp,"spotweld solid ID   %8d",ids[i]);
        if(fmax != NULL && fmax[i] > 0.)
        {
          fprintf(fp,"%13.5E",swlen[i]);
          fprintf(fp," fcmx=%s",tochar2(fmax[i],7));
          fprintf(fp," t=%s\n",tochar2(tfmax[i],8));
        }
        else if(ftime[i] > 0.0)
        {
          fprintf(fp,"               failure time=%12.4E\n",ftime[i]);
        }
        else
        {
          fprintf(fp,"%13.5E",swlen[i]);
          if(rmom != NULL)
            fprintf(fp,"%13.5E",rmom[i]);
          else
            fprintf(fp,"             ");
          if(torsion)
            fprintf(fp,"%13.5E\n",torsion[i]);
          else
            fprintf(fp,"\n");
        }
      } else if(type[i] == 4) {
        fprintf(fp,"nonnodal cnst  ID   %8d",ids[i]);
        if(fflag[i])
          fprintf(fp,"  failure time=%12.4E  %13.5E\n",ftime[i],emom[j]);
        else
          fprintf(fp,"                             %13.5E\n",emom[j]);
        j++;
      } else if(type[i] == 6) {
        fprintf(fp,"spotweld assmy ID   %8d",ids[i]);
        if(ftime[i] > 0.0)
          fprintf(fp,"               failure time=%12.4E\n",ftime[i]);
        else {
          fprintf(fp,"%13.5E",swlen[i]);
          if(rmom != NULL)
            fprintf(fp,"%13.5E",rmom[i]);
          else
            fprintf(fp,"             ");
          if(torsion)
            fprintf(fp,"%13.5E\n",torsion[i]);
          else
            fprintf(fp,"\n");
        }
      } else {
        fprintf(fp,"\n");
      }
    }
  }
  fclose(fp);
  if(rmom!=NULL) free(rmom);
  if(fmax!=NULL) free(fmax);
  if(tfmax!=NULL) free(tfmax);
  if(torsion!=NULL) free(torsion);
  if(nnnodal) free(emom);
  if(fp1!=NULL) free(fp1);
  if(fn1!=NULL) free(fn1);
  if(fp2!=NULL) free(fp2);
  if(fn2!=NULL) free(fn2);
  free(swlen);
  free(ftime);
  free(shear);
  free(axial);
  free(fflag);
  free(type);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  ABSTAT file
*/
int translate_abstat(int handle)
{
  int i,j,k,typid,num,filenum,state,nmatc;
  LSDA_Length length;
  char dirname[256];
  int *ids, *nmat, *matids, nmatids;
  float *v,*p,*ie,*din,*den,*dout,*tm,*gt,*sa,*r, *b, *ub, *doutp, *doutv;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;
  int idoutp=0;

  if (lsda_cd(handle,"/abstat/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_queryvar(handle,"../d000001/volume",&typid,&length,&filenum);
  if(typid < 0) return 0;
  num = length;

  ids  = (int *) malloc(num*sizeof(int));
  if(lsda_read(handle,LSDA_INT,"ids",0,num,ids) != num) {
    for(i=0; i<num; i++)
       ids[i] = i+1;
  }
  v    = (float *) malloc(num*sizeof(float));
  p    = (float *) malloc(num*sizeof(float));
  ie   = (float *) malloc(num*sizeof(float));
  din  = (float *) malloc(num*sizeof(float));
  den  = (float *) malloc(num*sizeof(float));
  dout = (float *) malloc(num*sizeof(float));
  tm   = (float *) malloc(num*sizeof(float));
  gt   = (float *) malloc(num*sizeof(float));
  sa   = (float *) malloc(num*sizeof(float));
  r    = (float *) malloc(num*sizeof(float));

  lsda_queryvar(handle,"/abstat/d000001/dm_dt_outp",&typid,&length,&filenum);
  if(length > 0) {
    doutp= (float *) malloc(num*sizeof(float));
    doutv= (float *) malloc(num*sizeof(float));
    idoutp=1;
  }
/*
  Check for (optional?) blockage output
*/
  lsda_queryvar(handle,"../metadata/mat_ids",&typid,&length,&filenum);
  if(typid < 0) {
     nmatids = 0;
  } else {
     nmatids = length;
     matids = (int *) malloc(nmatids*sizeof(int));
     lsda_read(handle,LSDA_INT,"mat_ids",0,nmatids,matids);

     lsda_queryvar(handle,"../metadata/mat_counts",&typid,&length,&filenum);
     nmatc = length;
     nmat = (int *) malloc(nmatc*sizeof(int));
     lsda_read(handle,LSDA_INT,"mat_counts",0,nmatc,nmat);

     b = (float *) malloc(nmatids*sizeof(float));
     ub = (float *) malloc(nmatids*sizeof(float));
  }
/*
  open file and write header
*/
  printf("Extracting ABSTAT data\n");
  sprintf(output_file,"%sabstat",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/abstat/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/abstat",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",           0,  1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"volume",         0,num,    v) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"pressure",       0,num,    p) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,num,   ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_in",       0,num,  din) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"density",        0,num,  den) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_out",      0,num, dout) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_mass",     0,num,   tm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"gas_temp",       0,num,   gt) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"surface_area",   0,num,   sa) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"reaction",       0,num,    r) != num) break;

    if(idoutp==1) {
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_outp",     0,num,doutp) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_outv",     0,num,doutv) != num) break;
    }

    if(nmatids > 0) {
      lsda_read(handle,LSDA_FLOAT,"blocked_area",      0,nmatids,b);
      lsda_read(handle,LSDA_FLOAT,"unblocked_area",    0,nmatids,ub);
    }
    fprintf(fp,"\n\n      time   airbag/cv #  volume        pressure    internal energy");
    fprintf(fp,"      dm/dt in      density dm/dt out  total mass gas temp. surface area");
    fprintf(fp,"   reaction dm/dt outp dm/dt outv\n");
    k=0;
    for(i=0; i<num; i++) {
      fprintf(fp,"%12.5E%8d%15.4E%15.4E%15.4E%15.4E%15.4E",
       time,ids[i],v[i],p[i],ie[i],din[i],den[i]);

      if(idoutp==1) {
        fprintf(fp,"%11.3E%11.3E%11.3E%11.3E%11.3E%11.3E%11.3E\n",
         dout[i],tm[i],gt[i],sa[i],r[i],doutp[i],doutv[i]);
      } else {
        fprintf(fp,"%11.3E%11.3E%11.3E%11.3E%11.3E\n",
         dout[i],tm[i],gt[i],sa[i],r[i]);
      }

      if(nmatids > 0) {
        for(j=0; j<nmat[i]; j++) {
          fprintf(fp,"   Material I.D.=%8d  Unblocked  Area =",matids[k]);
          fprintf(fp,"%12.4E    Blocked  Area=%12.4E\n",ub[k],b[k]);
          k=k+1;
        }
      }
    }
  }
  fclose(fp);
  free(r);
  free(sa);
  free(gt);
  free(tm);
  free(dout);
  if(idoutp==1) {
    free(doutp);
    free(doutv);
  }
  free(den);
  free(din);
  free(ie);
  free(p);
  free(v);
  free(ids);
  if(nmatids > 0) {
    free(ub);
    free(b);
    free(nmat);
    free(matids);
  }
  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  PABSTAT file - Particle Gas Dynamic
*/
int translate_abstat_cpm(int handle)
{
  int i,j,jj,k,kk,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256],dname[256];
  int *bag_id;
  int *nspec, *nparts, *part_id, *part_type, numtot, numspectot, numvent;
  float *v,*p,*ie,*din,*den,*dout,*tm,*gt,*sa,*r;
  float time;
  FILE *fp, *fp_tmp1, *fp_tmp2, *fp_tmp3;
  char output_file_tmp1[256], output_file_tmp2[256];
  LSDADir *dp = NULL;
/* more data for cpm chamber option */
  int nchm,ntchm,*nbag_c,*chm_iid,*chm_uid;
  float *c_v,*c_p,*c_ie,*c_din,*c_den,*c_dout,*c_tm,*c_gt,*c_sa,*c_r,*c_te;
  char output_file_tmp3[256];

/* airbag parts data */
  float *ppres, *ppor_leak, *pvent_leak, *parea_tot, *parea_unblocked, *ptemp;
  float *ppresp, *ppresm;
  float *pnt_spec;

  if (lsda_cd(handle,"/abstat_cpm/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_queryvar(handle,"../d000001/volume",&typid,&length,&filenum);
  if(typid < 0) return 0;
  num = length;

  lsda_queryvar(handle,"nbag_chamber",&typid,&length,&filenum);
  if(typid < 0)  nchm=0; else nchm=length;
  if(nchm>0) {
    nbag_c = (int *) malloc(nchm*sizeof(int));
    lsda_read(handle,LSDA_INT,"nbag_chamber",0,nchm,nbag_c);

    lsda_queryvar(handle,"chamber_iid",&typid,&length,&filenum);
    ntchm=length;
    chm_iid = (int *) malloc(ntchm*sizeof(int));
    chm_uid = (int *) malloc(ntchm*sizeof(int));
    lsda_read(handle,LSDA_INT,"chamber_iid",0,ntchm,chm_iid);
    lsda_read(handle,LSDA_INT,"chamber_uid",0,ntchm,chm_uid);

    lsda_queryvar(handle,"../d000001/chamber_data/volume",&typid,&length,&filenum);
    ntchm=length;
    c_v    = (float *) malloc(ntchm*sizeof(float));
    c_p    = (float *) malloc(ntchm*sizeof(float));
    c_ie   = (float *) malloc(ntchm*sizeof(float));
    c_din  = (float *) malloc(ntchm*sizeof(float));
    c_den  = (float *) malloc(ntchm*sizeof(float));
    c_dout = (float *) malloc(ntchm*sizeof(float));
    c_tm   = (float *) malloc(ntchm*sizeof(float));
    c_gt   = (float *) malloc(ntchm*sizeof(float));
    c_sa   = (float *) malloc(ntchm*sizeof(float));
    c_r    = (float *) malloc(ntchm*sizeof(float));
    c_te   = (float *) malloc(ntchm*sizeof(float));
  }

  bag_id  = (int *) malloc(num*sizeof(int));
  nspec   = (int *) malloc(num*sizeof(int));
  nparts  = (int *) malloc(num*sizeof(int));
  if(lsda_read(handle,LSDA_INT,"ids",0,num,bag_id) != num) {
    for(i=0; i<num; i++)
       bag_id[i] = i+1;
  }

  lsda_read(handle,LSDA_INT,"nspec", 0,num,nspec );
  
  lsda_read(handle,LSDA_INT,"nparts",0,num,nparts);
  numtot=0;
  for(i=0; i<num; i++) numtot += nparts[i];
  numspectot=0;
  for(i=0; i<num; i++) numspectot += nparts[i] * (nspec[i]+1);

  part_id   = (int *) malloc(numtot*sizeof(int));
  part_type = (int *) malloc(numtot*sizeof(int));
  lsda_read(handle,LSDA_INT,"pid"  ,0,numtot,part_id);
  lsda_read(handle,LSDA_INT,"ptype",0,numtot,part_type);

  v    = (float *) malloc(num*sizeof(float));
  p    = (float *) malloc(num*sizeof(float));
  ie   = (float *) malloc(num*sizeof(float));
  din  = (float *) malloc(num*sizeof(float));
  den  = (float *) malloc(num*sizeof(float));
  dout = (float *) malloc(num*sizeof(float));
  tm   = (float *) malloc(num*sizeof(float));
  gt   = (float *) malloc(num*sizeof(float));
  sa   = (float *) malloc(num*sizeof(float));
  r    = (float *) malloc(num*sizeof(float));
/*
  read in part and species data
*/
  ppres           = (float *) malloc(numtot*sizeof(float));
  ppor_leak       = (float *) malloc(numtot*sizeof(float));
  pvent_leak      = (float *) malloc(numtot*sizeof(float));
  parea_tot       = (float *) malloc(numtot*sizeof(float));
  parea_unblocked = (float *) malloc(numtot*sizeof(float));
  ptemp           = (float *) malloc(numtot*sizeof(float));
  ppresp          = (float *) malloc(numtot*sizeof(float));
  ppresm          = (float *) malloc(numtot*sizeof(float));

  pnt_spec        = (float *) malloc(numspectot*sizeof(float));

/*
  open file and write header
*/
  printf("Extracting PARTICLE ABSTAT_CPM data\n");
  sprintf(output_file,"%sabstat_cpm",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/abstat_cpm/metadata",fp);
  output_legend(handle,fp,1,1);

  if(nchm>0) {
    sprintf(output_file_tmp3,"%sabstat_chamber",output_path);
    fp_tmp3=fopen(output_file_tmp3,"w");
    write_message(fp_tmp3,output_file_tmp3);
    output_title(handle,"/abstat_cpm/metadata",fp_tmp3);
    output_legend(handle,fp_tmp3,1,1);
  }

  sprintf(output_file_tmp1,"%spartgas_partdata_p.txt",output_path);
  fp_tmp1=fopen(output_file_tmp1,"w");
  output_title(handle,"/abstat_cpm/metadata",fp_tmp1);
  output_legend(handle,fp_tmp1,1,1);
  fprintf(fp_tmp1,"        time\n PART ID    pressure    por_leak   vent_leak   part_area  unblk_area temperature\n            press s+    press s-\n\n");

  numvent=jj=kk=0;
  for(i=0; i<num; i++) {
    if(nspec[i]>0) for(k=0;k<nparts[i];k++) if(part_type[kk+k] >= 3) numvent++;
    kk += nparts[i];
    jj += nspec[i]+1;
  }

  if(numvent) {
  sprintf(output_file_tmp2,"%spartgas_partdata_sp.txt",output_path);
  fp_tmp2=fopen(output_file_tmp2,"w");
  output_title(handle,"/abstat_cpm/metadata",fp_tmp2);
  output_legend(handle,fp_tmp2,1,1);
  fprintf(fp_tmp2,"        time\n PART ID    dmout/dt\n\n");
  }

/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/abstat_cpm",dp,dname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",           0,  1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"volume",         0,num,    v) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"pressure",       0,num,    p) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,num,   ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_in",       0,num,  din) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"density",        0,num,  den) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_out",      0,num, dout) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_mass",     0,num,   tm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"gas_temp",       0,num,   gt) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"surface_area",   0,num,   sa) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"reaction",       0,num,    r) != num) break;
    fprintf(fp,"\n\n      time   airbag/cv #  volume        pressure    internal energy");
    fprintf(fp,"      dm/dt in      density dm/dt out  total mass gas temp. surface area");
    fprintf(fp,"   reaction\n");
    k=0;
    for(i=0; i<num; i++) {
      fprintf(fp,"%12.5E%8d%15.4E%15.4E%15.4E%15.4E%15.4E",
       time,bag_id[i],v[i],p[i],ie[i],din[i],den[i]);
      fprintf(fp,"%11.3E%11.3E%11.3E%11.3E%11.3E\n",
       dout[i],tm[i],gt[i],sa[i],r[i]);
    }

    sprintf(dirname,"/abstat_cpm/%s/bag_data",dname);
    lsda_queryvar(handle,dirname,&typid,&length,&filenum);
    if(typid != 0) break;
    lsda_cd(handle,dirname);
    lsda_read(handle,LSDA_FLOAT,"pressure",      0,numtot,ppres);
    lsda_read(handle,LSDA_FLOAT,"por_leak",      0,numtot,ppor_leak);
    lsda_read(handle,LSDA_FLOAT,"vent_leak",     0,numtot,pvent_leak);
    lsda_read(handle,LSDA_FLOAT,"area_tot",      0,numtot,parea_tot);
    lsda_read(handle,LSDA_FLOAT,"area_unblocked",0,numtot,parea_unblocked);
    lsda_read(handle,LSDA_FLOAT,"temperature",   0,numtot,ptemp);
    lsda_read(handle,LSDA_FLOAT,"pres+",         0,numtot,ppresp);
    lsda_read(handle,LSDA_FLOAT,"pres-",         0,numtot,ppresm);

    fprintf(fp_tmp1,"%12.4E\n",time);
    for(i=0; i<numtot; i++) {
      fprintf(fp_tmp1,"%8d%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n        %12.4E%12.4E\n",
       part_id[i],ppres[i],ppor_leak[i],pvent_leak[i],parea_tot[i],
       parea_unblocked[i],ptemp[i],ppresp[i],ppresm[i]);
    }

    if(numvent) {
      lsda_read(handle,LSDA_FLOAT,"nt_species",0,numspectot,pnt_spec);
      fprintf(fp_tmp2,"%12.4E\n",time);
      jj=kk=0;
      for(i=0; i<num; i++) {
        if(nspec[i]>0) {
          for(k=0; k<nparts[i]; k++) {
            if(part_type[kk+k] >= 3) {
              fprintf(fp_tmp2,"%8d%12.4E",part_id[kk+k],ptemp[kk+k]);
              for(j=0;j<=nspec[i];j++) fprintf(fp_tmp2,"%12.4E",pnt_spec[jj+j]);
              fprintf(fp_tmp2,"\n");
            }
          }
        }
        kk += nparts[i];
        jj += nspec[i]+1;
      }
    }
    if(nchm>0) {
      sprintf(dirname,"/abstat_cpm/%s/chamber_data",dname);
      lsda_queryvar(handle,dirname,&typid,&length,&filenum);
      if(typid != 0) break;
      lsda_cd(handle,dirname);

      if(lsda_read(handle,LSDA_FLOAT,"volume",         0,ntchm,   c_v) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"pressure",       0,ntchm,   c_p) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,ntchm,  c_ie) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"dm_dt_in",       0,ntchm, c_din) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"density",        0,ntchm, c_den) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"dm_dt_out",      0,ntchm,c_dout) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"total_mass",     0,ntchm,  c_tm) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"gas_temp",       0,ntchm,  c_gt) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"surface_area",   0,ntchm,  c_sa) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"reaction",       0,ntchm,   c_r) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"transE",         0,ntchm,  c_te) != ntchm) break;
      fprintf(fp_tmp3,"\n\n      time   airbag/cv #  volume        pressure    internal energy");
      fprintf(fp_tmp3,"      dm/dt in      density dm/dt out  total mass gas temp. surface area");
      fprintf(fp_tmp3,"   reaction         Trans E\n");

      for(i=0; i<ntchm; i++) {
        for(j=0; j<ntchm; j++) if(chm_iid[j]==i+1) { 
          fprintf(fp_tmp3,"%12.5E%8d%15.4E%15.4E%15.4E%15.4E%15.4E",
           time,chm_uid[j],c_v[i],c_p[i],c_ie[i],c_din[i],c_den[i]);
          fprintf(fp_tmp3,"%11.3E%11.3E%11.3E%11.3E%11.3E%11.3E\n",
           c_dout[i],c_tm[i],c_gt[i],c_sa[i],c_r[i],c_te[i]);
        }
      }
    }
  }
  fclose(fp);
  fclose(fp_tmp1);
  if(numvent) fclose(fp_tmp2);

  free(r);
  free(sa);
  free(gt);
  free(tm);
  free(dout);
  free(den);
  free(din);
  free(ie);
  free(p);
  free(v);

  free(ppres);
  free(ppor_leak);
  free(pvent_leak);
  free(parea_tot);
  free(parea_unblocked);
  free(ptemp);
  free(pnt_spec);

  free(part_type);
  free(part_id);
  free(nparts);
  free(nspec);
  free(bag_id);

  if(nchm>0) { 
    fclose(fp_tmp3);

    free(c_te);
    free(c_r);
    free(c_sa);
    free(c_gt);
    free(c_tm);
    free(c_dout);
    free(c_den);
    free(c_din);
    free(c_ie);
    free(c_p);
    free(c_v);
    free(chm_uid);
    free(chm_iid);
    free(nbag_c);
  }

  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  ABSTAT PBlast file - Particle Blast
*/
int translate_abstat_pbm(int handle)
{
  int i,j,jj,k,kk,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256],dname[256];
  int *bag_id;
  int *nparts, *part_id, *part_type, numtot;
  float *air_ie, *air_tr, *he_ie, *he_tr, *out_ie, *out_tr;
  float time;
  FILE *fp;
  char output_file_tmp1[256], output_file_tmp2[256];
  LSDADir *dp = NULL;

  float *c_v,*c_p,*c_ie,*c_din,*c_den,*c_dout,*c_tm,*c_gt,*c_sa,*c_r,*c_te;
  char output_file_tmp3[256];

  float *pp_air,*pp_he,*pp_result,*area,*fx_air,*fy_air,*fz_air,*fx_he,*fy_he,*fz_he,*fx_result,*fy_result,*fz_result;

  if (lsda_cd(handle,"/abstat_pbm/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_read(handle,LSDA_INT,"nbag",0,1,&num);

  bag_id  = (int *) malloc(num*sizeof(int));
  nparts  = (int *) malloc(num*sizeof(int));
  if(lsda_read(handle,LSDA_INT,"ids",0,num,bag_id) != num) {
    for(i=0; i<num; i++)
       bag_id[i] = i+1;
  }

  lsda_read(handle,LSDA_INT,"nparts",0,num,nparts);

  numtot=0;
  for(i=0; i<num; i++) numtot += nparts[i];

  part_id   = (int *) malloc(numtot*sizeof(int));
  lsda_read(handle,LSDA_INT,"pid"  ,0,numtot,part_id);

  air_ie    = (float *) malloc(num*sizeof(float));
  air_tr    = (float *) malloc(num*sizeof(float));
  he_ie     = (float *) malloc(num*sizeof(float));
  he_tr     = (float *) malloc(num*sizeof(float));
  out_ie    = (float *) malloc(num*sizeof(float));
  out_tr    = (float *) malloc(num*sizeof(float));
/*
  read in part and species data
*/
  pp_air          = (float *) malloc(numtot*sizeof(float));
  pp_he           = (float *) malloc(numtot*sizeof(float));
  pp_result       = (float *) malloc(numtot*sizeof(float));
  area            = (float *) malloc(numtot*sizeof(float));
  fx_air          = (float *) malloc(numtot*sizeof(float));
  fy_air          = (float *) malloc(numtot*sizeof(float));
  fz_air          = (float *) malloc(numtot*sizeof(float));
  fx_he           = (float *) malloc(numtot*sizeof(float));
  fy_he           = (float *) malloc(numtot*sizeof(float));
  fz_he           = (float *) malloc(numtot*sizeof(float));
  fx_result       = (float *) malloc(numtot*sizeof(float));
  fy_result       = (float *) malloc(numtot*sizeof(float));
  fz_result       = (float *) malloc(numtot*sizeof(float));

/*
  open file and write header
*/
  printf("Extracting PARTICLE ABSTAT_PBM data\n");
  sprintf(output_file,"%sabstat_pbm",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/abstat_pbm/metadata",fp);
  output_legend(handle,fp,1,1);
  fprintf(fp,"\n Total number Particle Blast defined: %d\n                  Total number parts: %d\n\n",num,numtot);

/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/abstat_pbm",dp,dname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",                      0,  1,  &time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"air_inter_e",               0,num, air_ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"air_trans_e",               0,num, air_tr) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"detonation_product_inter_e",0,num,  he_ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"detonation_product_trans_e",0,num,  he_tr) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"outside_domain_inter_e",    0,num, out_ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"outside_domain_trans_e",    0,num, out_tr) != num) break;

    fprintf(fp,"\n\n      time : %15.4E\n",time);
    fprintf(fp,"\n\nPBlast #         Air IE         Air TE          HE IE          HE TE         Out IE         Out TE\n");

    for(i=0; i<num; i++) {
      fprintf(fp,"%8d%15.4E%15.4E%15.4E%15.4E%15.4E%15.4E\n",
       bag_id[i],air_ie[i],air_tr[i],he_ie[i],he_tr[i],out_ie[i],out_tr[i]);
    }

    sprintf(dirname,"/abstat_pbm/%s/bag_data",dname);
    lsda_queryvar(handle,dirname,&typid,&length,&filenum);
    if(typid != 0) break;
    lsda_cd(handle,dirname);
    lsda_read(handle,LSDA_FLOAT,"pressure_air",         0,numtot,pp_air);
    lsda_read(handle,LSDA_FLOAT,"pressure_det_products",0,numtot,pp_he);
    lsda_read(handle,LSDA_FLOAT,"pressure_resultant",   0,numtot,pp_result);
    lsda_read(handle,LSDA_FLOAT,"surface_area",         0,numtot,area);
    lsda_read(handle,LSDA_FLOAT,"x_force_air",          0,numtot,fx_air);
    lsda_read(handle,LSDA_FLOAT,"y_force_air",          0,numtot,fy_air);
    lsda_read(handle,LSDA_FLOAT,"z_force_air",          0,numtot,fz_air);
    lsda_read(handle,LSDA_FLOAT,"x_force_det_products", 0,numtot,fx_he);
    lsda_read(handle,LSDA_FLOAT,"y_force_det_products", 0,numtot,fy_he);
    lsda_read(handle,LSDA_FLOAT,"z_force_det_products", 0,numtot,fz_he);
    lsda_read(handle,LSDA_FLOAT,"x_force_resultant",    0,numtot,fx_result);
    lsda_read(handle,LSDA_FLOAT,"y_force_resultant",    0,numtot,fy_result);
    lsda_read(handle,LSDA_FLOAT,"z_force_resultant",    0,numtot,fz_result);

    fprintf(fp,"\n\n  Part #       Air P        HE P Resultant P        Area      Air Fx      AIR Fy      Air Fz\n");
    fprintf(fp,    "               HE Fx       HE Fy       HE Fz        R Fx        R Fy        R Fz\n");

    for(i=0; i<numtot; i++) {
      fprintf(fp,"%8d%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n        %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
       part_id[i],pp_air[i],pp_he[i],pp_result[i],area[i],fx_air[i],fy_air[i],fz_air[i],
       fx_he[i],fy_he[i],fy_he[i],fx_result[i],fy_result[i],fz_result[i]);
    }
  }
  fclose(fp);

  free(pp_air);
  free(pp_he);
  free(pp_result);
  free(area);
  free(fx_air);
  free(fy_air);
  free(fz_air);
  free(fx_he);
  free(fy_he);
  free(fz_he);
  free(fx_result);
  free(fy_result);
  free(fz_result);

  free(air_ie);
  free(air_tr);
  free(he_ie);
  free(he_tr);
  free(out_ie);
  free(out_tr);
  free(bag_id);
  free(part_id);

  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  CPM_SENSOR file
*/
int translate_cpm_sensor(int handle)
{
  int i,state,nsensor,revision,newformat;
  LSDA_Length length;
  FILE *fp;
  float time,*dens,*pres,*temp,*velx,*vely,*velz,*velr,*xpos,*ypos,*zpos,*part;
  char dirname[256];
  char name[15],*rev;
  LSDADir *dp = NULL;
  if (lsda_cd(handle,"/cpm_sensor/metadata") == -1) return 0;
/*
  Read metadata
*/
  rev = (char *)malloc(10);
  lsda_read(handle,LSDA_I1,"../metadata/revision",0,10,rev);
  revision = atoi(rev);
  free(rev);
  lsda_read(handle,LSDA_INT,"../metadata/n_bags",0,1,&newformat);

  if (revision > 48400) {
    if (newformat != 0) {
      translate_cpm_sensor_new(handle);
      return 0;
    }
  }
  lsda_read(handle,LSDA_INT,"../metadata/nsensor",0,1,&nsensor);
  printf("Extracting CPM_SENSOR data\n");

  velr    = (float *) malloc(nsensor*sizeof(float));
  velx    = (float *) malloc(nsensor*sizeof(float));
  vely    = (float *) malloc(nsensor*sizeof(float));
  velz    = (float *) malloc(nsensor*sizeof(float));
  temp    = (float *) malloc(nsensor*sizeof(float));
  pres    = (float *) malloc(nsensor*sizeof(float));
  dens    = (float *) malloc(nsensor*sizeof(float));
  xpos    = (float *) malloc(nsensor*sizeof(float));
  ypos    = (float *) malloc(nsensor*sizeof(float));
  zpos    = (float *) malloc(nsensor*sizeof(float));
  part    = (float *) malloc(nsensor*sizeof(float));
/*
  open file and write header
*/
  sprintf(output_file,"%scpm_sensor",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  fprintf(fp,"Airbag Particle sensor file\n");
  fprintf(fp,"%5d%5d\n",nsensor,16);
  fprintf(fp,"         time\n");
  fprintf(fp,"            x            y            z\n");
  fprintf(fp,"           vx           vy           vz\n");
  fprintf(fp,"         pres         dens         temp        npart\n\n\n");
  
/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/cpm_sensor",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_velx",0,nsensor,velx) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_vely",0,nsensor,vely) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_velz",0,nsensor,velz) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_velr",0,nsensor,velr) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"temp",0,nsensor,temp) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"rho",0,nsensor,dens) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"pressure",0,nsensor,pres) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"sensor_x",0,nsensor,xpos) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"sensor_y",0,nsensor,ypos) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"sensor_z",0,nsensor,zpos) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"npart",0,nsensor,part) != nsensor) break;

    fprintf(fp,"%13.5E\n",time);
    for (i=0; i<nsensor; i++) { 
      fprintf(fp,"%13.5E%13.5E%13.5E\n",xpos[i],ypos[i],zpos[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E\n",velx[i],vely[i],velz[i],velr[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13d\n",pres[i],dens[i],temp[i],(int) part[i]);
    }
  }
  fclose(fp);

  free(part);
  free(zpos);
  free(ypos);
  free(xpos);
  free(dens);
  free(pres);
  free(temp);
  free(velz);
  free(vely);
  free(velx);
  free(velr);

  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  CPM_SENSOR file with new format
*/
int translate_cpm_sensor_new(int handle)
{
  int i,j,k,state;
  FILE *fp;
  char dirname[256];
  int npartgas,nhcpmsor,icpmsor_nd,ntotsor,nin;
  int *ids,*n_seg;

  float *dens,*pres,*temp,*velr,*velx,*vely,*velz,*xpos,*ypos,*zpos;
  float time;
  int   m,mi,isensor;

  LSDADir *dp = NULL;
  if (lsda_cd(handle,"/cpm_sensor/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_read(handle,LSDA_INT,"n_bags",0,1,&npartgas);
  lsda_read(handle,LSDA_INT,"n_set_sensor",0,1,&nhcpmsor);
  lsda_read(handle,LSDA_INT,"n_struc",0,1,&icpmsor_nd);
  lsda_read(handle,LSDA_INT,"n_sensor",0,1,&ntotsor);

  nin   = npartgas*nhcpmsor;
  ids   = (int *) malloc(ntotsor*sizeof(int));
  n_seg = (int *) malloc(nin*sizeof(int));
  lsda_read(handle,LSDA_INT,"id_sensor",0,ntotsor,ids);
  lsda_read(handle,LSDA_INT,"n_seg_in_each_set",0,nin,n_seg);

  printf("Extracting CPM_SENSOR from new database\n");
/*
  open file and write header
*/
  sprintf(output_file,"%strhist_cpm_sensor",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;

  fprintf(fp,"     CPM sensors output\n");
  fprintf(fp,"     Number of sensors:%5d\n\n\n",ntotsor);
  fprintf(fp,"        id             x             y             z          velx          vely\n");
  fprintf(fp,"                    velx          velr          temp          dens          pres\n");
/*
  Loop through time states and write each one
*/
  velr    = (float *) malloc(ntotsor*sizeof(float));
  velx    = (float *) malloc(ntotsor*sizeof(float));
  vely    = (float *) malloc(ntotsor*sizeof(float));
  velz    = (float *) malloc(ntotsor*sizeof(float));
  temp    = (float *) malloc(ntotsor*sizeof(float));
  pres    = (float *) malloc(ntotsor*sizeof(float));
  dens    = (float *) malloc(ntotsor*sizeof(float));
  xpos    = (float *) malloc(ntotsor*sizeof(float));
  ypos    = (float *) malloc(ntotsor*sizeof(float));
  zpos    = (float *) malloc(ntotsor*sizeof(float));

  for(state=1; (dp = next_dir(handle,"/cpm_sensor",dp,dirname)) != NULL; state++) {
     if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
     fprintf(fp,"   time=%13.5E\n",time);

     lsda_read(handle,LSDA_FLOAT,"ave_velx",0,ntotsor,velx);
     lsda_read(handle,LSDA_FLOAT,"ave_vely",0,ntotsor,vely);
     lsda_read(handle,LSDA_FLOAT,"ave_velz",0,ntotsor,velz);
     lsda_read(handle,LSDA_FLOAT,"ave_velr",0,ntotsor,velr);
     lsda_read(handle,LSDA_FLOAT,"temp",0,ntotsor,temp);
     lsda_read(handle,LSDA_FLOAT,"rho",0,ntotsor,dens);
     lsda_read(handle,LSDA_FLOAT,"pressure",0,ntotsor,pres);
     lsda_read(handle,LSDA_FLOAT,"sensor_x",0,ntotsor,xpos);
     lsda_read(handle,LSDA_FLOAT,"sensor_y",0,ntotsor,ypos);
     lsda_read(handle,LSDA_FLOAT,"sensor_z",0,ntotsor,zpos);

     k=m=0;
     for(i=1;i<=npartgas;i++) { 
       for(j=1;j<=nhcpmsor;j++) { 
         if(n_seg[k] != 0) {
           fprintf(fp,"          Bag ID, Sensor set:%10d%10d\n",i,j);
           for(isensor=0; isensor<n_seg[k];isensor++){
             mi=m+isensor;
             fprintf(fp,"%10d%14.4e%14.4e%14.4e%14.4e%14.4e\n",
                     ids[mi],xpos[mi],ypos[mi],zpos[mi],velx[mi],vely[mi]);
             fprintf(fp,"          %14.4e%14.4e%14.4e%14.4e%14.4e\n",
                     velz[mi],velr[mi],temp[mi],dens[mi],pres[mi]);
           } 
           fprintf(fp,"\n"); m += n_seg[k];
         }
         k++;
       }
     }
  }
  fclose(fp);
  printf("      %d states extracted\n",state-1);

  free(velr);
  free(velx);
  free(vely);
  free(velz);
  free(temp);
  free(pres);
  free(dens);
  free(xpos);
  free(ypos);
  free(zpos);

  return 0;
}
/*
  PG_STAT file - Particle general 
*/
int translate_pgstat(int handle)
{
  int i,j,jj,k,kk,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256],dname[256];
  int *bag_id;
  int *nspec, *nparts, *part_id, *part_type, numtot, numspectot, numvent;
  float *v,*p,*ie,*din,*den,*dout,*tm,*gt,*sa,*r;
  float time;
  FILE *fp, *fp_tmp1, *fp_tmp2, *fp_tmp3;
  char output_file_tmp1[256], output_file_tmp2[256];
  LSDADir *dp = NULL;
/* more data for PG chamber option */
  int nchm,ntchm,*nbag_c,*chm_iid,*chm_uid;
  float *c_v,*c_p,*c_ie,*c_din,*c_den,*c_dout,*c_tm,*c_gt,*c_sa,*c_r,*c_te;
  char output_file_tmp3[256];

/* airbag parts data */
  float *ppres, *ppor_leak, *pvent_leak, *parea_tot, *parea_unblocked, *ptemp;
  float *ppresp, *ppresm;
  float *pnt_spec;

  if (lsda_cd(handle,"/pg_stat/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_queryvar(handle,"../d000001/volume",&typid,&length,&filenum);
  if(typid < 0) return 0;
  num = length;

  lsda_queryvar(handle,"nbag_chamber",&typid,&length,&filenum);
  if(typid < 0)  nchm=0; else nchm=length;
  if(nchm>0) {
    nbag_c = (int *) malloc(nchm*sizeof(int));
    lsda_read(handle,LSDA_INT,"nbag_chamber",0,nchm,nbag_c);

    lsda_queryvar(handle,"chamber_iid",&typid,&length,&filenum);
    ntchm=length;
    chm_iid = (int *) malloc(ntchm*sizeof(int));
    chm_uid = (int *) malloc(ntchm*sizeof(int));
    lsda_read(handle,LSDA_INT,"chamber_iid",0,ntchm,chm_iid);
    lsda_read(handle,LSDA_INT,"chamber_uid",0,ntchm,chm_uid);

    lsda_queryvar(handle,"../d000001/chamber_data/volume",&typid,&length,&filenum);
    ntchm=length;
    c_v    = (float *) malloc(ntchm*sizeof(float));
    c_p    = (float *) malloc(ntchm*sizeof(float));
    c_ie   = (float *) malloc(ntchm*sizeof(float));
    c_din  = (float *) malloc(ntchm*sizeof(float));
    c_den  = (float *) malloc(ntchm*sizeof(float));
    c_dout = (float *) malloc(ntchm*sizeof(float));
    c_tm   = (float *) malloc(ntchm*sizeof(float));
    c_gt   = (float *) malloc(ntchm*sizeof(float));
    c_sa   = (float *) malloc(ntchm*sizeof(float));
    c_r    = (float *) malloc(ntchm*sizeof(float));
    c_te   = (float *) malloc(ntchm*sizeof(float));
  }

  bag_id  = (int *) malloc(num*sizeof(int));
  nspec   = (int *) malloc(num*sizeof(int));
  nparts  = (int *) malloc(num*sizeof(int));
  if(lsda_read(handle,LSDA_INT,"ids",0,num,bag_id) != num) {
    for(i=0; i<num; i++)
       bag_id[i] = i+1;
  }

  lsda_read(handle,LSDA_INT,"nspec", 0,num,nspec );
  
  lsda_read(handle,LSDA_INT,"nparts",0,num,nparts);
  numtot=0;
  for(i=0; i<num; i++) numtot += nparts[i];
  numspectot=0;
  for(i=0; i<num; i++) numspectot += nparts[i] * (nspec[i]+1);

  part_id   = (int *) malloc(numtot*sizeof(int));
  part_type = (int *) malloc(numtot*sizeof(int));
  lsda_read(handle,LSDA_INT,"pid"  ,0,numtot,part_id);
  lsda_read(handle,LSDA_INT,"ptype",0,numtot,part_type);

  v    = (float *) malloc(num*sizeof(float));
  p    = (float *) malloc(num*sizeof(float));
  ie   = (float *) malloc(num*sizeof(float));
  din  = (float *) malloc(num*sizeof(float));
  den  = (float *) malloc(num*sizeof(float));
  dout = (float *) malloc(num*sizeof(float));
  tm   = (float *) malloc(num*sizeof(float));
  gt   = (float *) malloc(num*sizeof(float));
  sa   = (float *) malloc(num*sizeof(float));
  r    = (float *) malloc(num*sizeof(float));
/*
  read in part and species data
*/
  ppres           = (float *) malloc(numtot*sizeof(float));
  ppor_leak       = (float *) malloc(numtot*sizeof(float));
  pvent_leak      = (float *) malloc(numtot*sizeof(float));
  parea_tot       = (float *) malloc(numtot*sizeof(float));
  parea_unblocked = (float *) malloc(numtot*sizeof(float));
  ptemp           = (float *) malloc(numtot*sizeof(float));
  ppresp          = (float *) malloc(numtot*sizeof(float));
  ppresm          = (float *) malloc(numtot*sizeof(float));

  pnt_spec        = (float *) malloc(numspectot*sizeof(float));

/*
  open file and write header
*/
  printf("Extracting PARTICLE GENERAL PG_STAT data\n");
  sprintf(output_file,"%spg_stat",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/pg_stat/metadata",fp);
  output_legend(handle,fp,1,1);

  if(nchm>0) {
    sprintf(output_file_tmp3,"%spgstat_chamber",output_path);
    fp_tmp3=fopen(output_file_tmp3,"w");
    write_message(fp_tmp3,output_file_tmp3);
    output_title(handle,"/pg_stat/metadata",fp_tmp3);
    output_legend(handle,fp_tmp3,1,1);
  }

  sprintf(output_file_tmp1,"%spg_partdata_p.txt",output_path);
  fp_tmp1=fopen(output_file_tmp1,"w");
  output_title(handle,"/pg_stat/metadata",fp_tmp1);
  output_legend(handle,fp_tmp1,1,1);
  fprintf(fp_tmp1,"        time\n PART ID    pressure    por_leak   vent_leak   part_area  unblk_area temperature\n            press s+    press s-\n\n");

  numvent=jj=kk=0;
  for(i=0; i<num; i++) {
    if(nspec[i]>0) for(k=0;k<nparts[i];k++) if(part_type[kk+k] >= 3) numvent++;
    kk += nparts[i];
    jj += nspec[i]+1;
  }

  if(numvent) {
  sprintf(output_file_tmp2,"%spg_partdata_sp.txt",output_path);
  fp_tmp2=fopen(output_file_tmp2,"w");
  output_title(handle,"/pg_stat/metadata",fp_tmp2);
  output_legend(handle,fp_tmp2,1,1);
  fprintf(fp_tmp2,"        time\n PART ID    dmout/dt\n\n");
  }

/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/pg_stat",dp,dname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",           0,  1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"volume",         0,num,    v) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"pressure",       0,num,    p) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,num,   ie) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_in",       0,num,  din) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"density",        0,num,  den) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dm_dt_out",      0,num, dout) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_mass",     0,num,   tm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"gas_temp",       0,num,   gt) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"surface_area",   0,num,   sa) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"reaction",       0,num,    r) != num) break;
    fprintf(fp,"\n\n      time   airbag/cv #  volume        pressure    internal energy");
    fprintf(fp,"      dm/dt in      density dm/dt out  total mass gas temp. surface area");
    fprintf(fp,"   reaction\n");
    k=0;
    for(i=0; i<num; i++) {
      fprintf(fp,"%12.5E%8d%15.4E%15.4E%15.4E%15.4E%15.4E",
       time,bag_id[i],v[i],p[i],ie[i],din[i],den[i]);
      fprintf(fp,"%11.3E%11.3E%11.3E%11.3E%11.3E\n",
       dout[i],tm[i],gt[i],sa[i],r[i]);
    }

    sprintf(dirname,"/pg_stat/%s/bag_data",dname);
    lsda_queryvar(handle,dirname,&typid,&length,&filenum);
    if(typid != 0) break;
    lsda_cd(handle,dirname);
    lsda_read(handle,LSDA_FLOAT,"pressure",      0,numtot,ppres);
    lsda_read(handle,LSDA_FLOAT,"por_leak",      0,numtot,ppor_leak);
    lsda_read(handle,LSDA_FLOAT,"vent_leak",     0,numtot,pvent_leak);
    lsda_read(handle,LSDA_FLOAT,"area_tot",      0,numtot,parea_tot);
    lsda_read(handle,LSDA_FLOAT,"area_unblocked",0,numtot,parea_unblocked);
    lsda_read(handle,LSDA_FLOAT,"temperature",   0,numtot,ptemp);
    lsda_read(handle,LSDA_FLOAT,"pres+",         0,numtot,ppresp);
    lsda_read(handle,LSDA_FLOAT,"pres-",         0,numtot,ppresm);

    fprintf(fp_tmp1,"%12.4E\n",time);
    for(i=0; i<numtot; i++) {
      fprintf(fp_tmp1,"%8d%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n        %12.4E%12.4E\n",
       part_id[i],ppres[i],ppor_leak[i],pvent_leak[i],parea_tot[i],
       parea_unblocked[i],ptemp[i],ppresp[i],ppresm[i]);
    }

    if(numvent) {
      lsda_read(handle,LSDA_FLOAT,"nt_species",0,numspectot,pnt_spec);
      fprintf(fp_tmp2,"%12.4E\n",time);
      jj=kk=0;
      for(i=0; i<num; i++) {
        if(nspec[i]>0) {
          for(k=0; k<nparts[i]; k++) {
            if(part_type[kk+k] >= 3) {
              fprintf(fp_tmp2,"%8d%12.4E",part_id[kk+k],ptemp[kk+k]);
              for(j=0;j<=nspec[i];j++) fprintf(fp_tmp2,"%12.4E",pnt_spec[jj+j]);
              fprintf(fp_tmp2,"\n");
            }
          }
        }
        kk += nparts[i];
        jj += nspec[i]+1;
      }
    }
    if(nchm>0) {
      sprintf(dirname,"/pg_stat/%s/chamber_data",dname);
      lsda_queryvar(handle,dirname,&typid,&length,&filenum);
      if(typid != 0) break;
      lsda_cd(handle,dirname);

      if(lsda_read(handle,LSDA_FLOAT,"volume",         0,ntchm,   c_v) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"pressure",       0,ntchm,   c_p) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"internal_energy",0,ntchm,  c_ie) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"dm_dt_in",       0,ntchm, c_din) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"density",        0,ntchm, c_den) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"dm_dt_out",      0,ntchm,c_dout) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"total_mass",     0,ntchm,  c_tm) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"gas_temp",       0,ntchm,  c_gt) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"surface_area",   0,ntchm,  c_sa) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"reaction",       0,ntchm,   c_r) != ntchm) break;
      if(lsda_read(handle,LSDA_FLOAT,"transE",         0,ntchm,  c_te) != ntchm) break;
      fprintf(fp_tmp3,"\n\n      time   airbag/cv #  volume        pressure    internal energy");
      fprintf(fp_tmp3,"      dm/dt in      density dm/dt out  total mass gas temp. surface area");
      fprintf(fp_tmp3,"   reaction         Trans E\n");

      for(i=0; i<ntchm; i++) {
        for(j=0; j<ntchm; j++) if(chm_iid[j]==i+1) { 
          fprintf(fp_tmp3,"%12.5E%8d%15.4E%15.4E%15.4E%15.4E%15.4E",
           time,chm_uid[j],c_v[i],c_p[i],c_ie[i],c_din[i],c_den[i]);
          fprintf(fp_tmp3,"%11.3E%11.3E%11.3E%11.3E%11.3E%11.3E\n",
           c_dout[i],c_tm[i],c_gt[i],c_sa[i],c_r[i],c_te[i]);
        }
      }
    }
  }
  fclose(fp);
  fclose(fp_tmp1);
  if(numvent) fclose(fp_tmp2);

  free(r);
  free(sa);
  free(gt);
  free(tm);
  free(dout);
  free(den);
  free(din);
  free(ie);
  free(p);
  free(v);

  free(ppres);
  free(ppor_leak);
  free(pvent_leak);
  free(parea_tot);
  free(parea_unblocked);
  free(ptemp);
  free(pnt_spec);

  free(part_type);
  free(part_id);
  free(nparts);
  free(nspec);
  free(bag_id);

  if(nchm>0) { 
    fclose(fp_tmp3);

    free(c_te);
    free(c_r);
    free(c_sa);
    free(c_gt);
    free(c_tm);
    free(c_dout);
    free(c_den);
    free(c_din);
    free(c_ie);
    free(c_p);
    free(c_v);
    free(chm_uid);
    free(chm_iid);
    free(nbag_c);
  }

  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  PG_SENSOR file
*/
int translate_pg_sensor(int handle)
{
  int i,state,nsensor,revision,newformat;
  LSDA_Length length;
  FILE *fp;
  float time,*dens,*pres,*temp,*velx,*vely,*velz,*velr,*xpos,*ypos,*zpos,*part;
  char dirname[256];
  char name[15],*rev;
  LSDADir *dp = NULL;
  if (lsda_cd(handle,"/pg_sensor/metadata") == -1) return 0;
/*
  Read metadata
*/
  rev = (char *)malloc(10);
  lsda_read(handle,LSDA_I1,"../metadata/revision",0,10,rev);
  revision = atoi(rev);
  free(rev);
  lsda_read(handle,LSDA_INT,"../metadata/n_bags",0,1,&newformat);

  if (revision > 48400) {
    if (newformat != 0) {
      translate_pg_sensor_new(handle);
      return 0;
    }
  }
  lsda_read(handle,LSDA_INT,"../metadata/nsensor",0,1,&nsensor);
  printf("Extracting PG_SENSOR data\n");

  velr    = (float *) malloc(nsensor*sizeof(float));
  velx    = (float *) malloc(nsensor*sizeof(float));
  vely    = (float *) malloc(nsensor*sizeof(float));
  velz    = (float *) malloc(nsensor*sizeof(float));
  temp    = (float *) malloc(nsensor*sizeof(float));
  pres    = (float *) malloc(nsensor*sizeof(float));
  dens    = (float *) malloc(nsensor*sizeof(float));
  xpos    = (float *) malloc(nsensor*sizeof(float));
  ypos    = (float *) malloc(nsensor*sizeof(float));
  zpos    = (float *) malloc(nsensor*sizeof(float));
  part    = (float *) malloc(nsensor*sizeof(float));
/*
  open file and write header
*/
  sprintf(output_file,"%spg_sensor",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  fprintf(fp,"Particle general sensor file\n");
  fprintf(fp,"%5d%5d\n",nsensor,16);
  fprintf(fp,"         time\n");
  fprintf(fp,"            x            y            z\n");
  fprintf(fp,"           vx           vy           vz\n");
  fprintf(fp,"         pres         dens         temp        npart\n\n\n");
  
/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/pg_sensor",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_velx",0,nsensor,velx) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_vely",0,nsensor,vely) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_velz",0,nsensor,velz) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"ave_velr",0,nsensor,velr) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"temp",0,nsensor,temp) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"rho",0,nsensor,dens) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"pressure",0,nsensor,pres) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"sensor_x",0,nsensor,xpos) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"sensor_y",0,nsensor,ypos) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"sensor_z",0,nsensor,zpos) != nsensor) break;
    if(lsda_read(handle,LSDA_FLOAT,"npart",0,nsensor,part) != nsensor) break;

    fprintf(fp,"%13.5E\n",time);
    for (i=0; i<nsensor; i++) { 
      fprintf(fp,"%13.5E%13.5E%13.5E\n",xpos[i],ypos[i],zpos[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E\n",velx[i],vely[i],velz[i],velr[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13d\n",pres[i],dens[i],temp[i],(int) part[i]);
    }
  }
  fclose(fp);

  free(part);
  free(zpos);
  free(ypos);
  free(xpos);
  free(dens);
  free(pres);
  free(temp);
  free(velz);
  free(vely);
  free(velx);
  free(velr);

  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  PG_SENSOR file with new format
*/
int translate_pg_sensor_new(int handle)
{
  int i,j,k,state;
  FILE *fp;
  char dirname[256];
  int npartgas,nhcpmsor,icpmsor_nd,ntotsor,nin;
  int *ids,*n_seg;

  float *dens,*pres,*temp,*velr,*velx,*vely,*velz,*xpos,*ypos,*zpos;
  float time;
  int   m,mi,isensor;

  LSDADir *dp = NULL;
  if (lsda_cd(handle,"/pg_sensor/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_read(handle,LSDA_INT,"n_bags",0,1,&npartgas);
  lsda_read(handle,LSDA_INT,"n_set_sensor",0,1,&nhcpmsor);
  lsda_read(handle,LSDA_INT,"n_struc",0,1,&icpmsor_nd);
  lsda_read(handle,LSDA_INT,"n_sensor",0,1,&ntotsor);

  nin   = npartgas*nhcpmsor;
  ids   = (int *) malloc(ntotsor*sizeof(int));
  n_seg = (int *) malloc(nin*sizeof(int));
  lsda_read(handle,LSDA_INT,"id_sensor",0,ntotsor,ids);
  lsda_read(handle,LSDA_INT,"n_seg_in_each_set",0,nin,n_seg);

  printf("Extracting PG_SENSOR from new database\n");
/*
  open file and write header
*/
  sprintf(output_file,"%strhist_pg_sensor",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;

  fprintf(fp,"     CPM sensors output\n");
  fprintf(fp,"     Number of sensors:%5d\n\n\n",ntotsor);
  fprintf(fp,"        id             x             y             z          velx          vely\n");
  fprintf(fp,"                    velx          velr          temp          dens          pres\n");
/*
  Loop through time states and write each one
*/
  velr    = (float *) malloc(ntotsor*sizeof(float));
  velx    = (float *) malloc(ntotsor*sizeof(float));
  vely    = (float *) malloc(ntotsor*sizeof(float));
  velz    = (float *) malloc(ntotsor*sizeof(float));
  temp    = (float *) malloc(ntotsor*sizeof(float));
  pres    = (float *) malloc(ntotsor*sizeof(float));
  dens    = (float *) malloc(ntotsor*sizeof(float));
  xpos    = (float *) malloc(ntotsor*sizeof(float));
  ypos    = (float *) malloc(ntotsor*sizeof(float));
  zpos    = (float *) malloc(ntotsor*sizeof(float));

  for(state=1; (dp = next_dir(handle,"/pg_sensor",dp,dirname)) != NULL; state++) {
     if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
     fprintf(fp,"   time=%13.5E\n",time);

     lsda_read(handle,LSDA_FLOAT,"ave_velx",0,ntotsor,velx);
     lsda_read(handle,LSDA_FLOAT,"ave_vely",0,ntotsor,vely);
     lsda_read(handle,LSDA_FLOAT,"ave_velz",0,ntotsor,velz);
     lsda_read(handle,LSDA_FLOAT,"ave_velr",0,ntotsor,velr);
     lsda_read(handle,LSDA_FLOAT,"temp",0,ntotsor,temp);
     lsda_read(handle,LSDA_FLOAT,"rho",0,ntotsor,dens);
     lsda_read(handle,LSDA_FLOAT,"pressure",0,ntotsor,pres);
     lsda_read(handle,LSDA_FLOAT,"sensor_x",0,ntotsor,xpos);
     lsda_read(handle,LSDA_FLOAT,"sensor_y",0,ntotsor,ypos);
     lsda_read(handle,LSDA_FLOAT,"sensor_z",0,ntotsor,zpos);

     k=m=0;
     for(i=1;i<=npartgas;i++) { 
       for(j=1;j<=nhcpmsor;j++) { 
         if(n_seg[k] != 0) {
           fprintf(fp,"          Bag ID, Sensor set:%10d%10d\n",i,j);
           for(isensor=0; isensor<n_seg[k];isensor++){
             mi=m+isensor;
             fprintf(fp,"%10d%14.4e%14.4e%14.4e%14.4e%14.4e\n",
                     ids[mi],xpos[mi],ypos[mi],zpos[mi],velx[mi],vely[mi]);
             fprintf(fp,"          %14.4e%14.4e%14.4e%14.4e%14.4e\n",
                     velz[mi],velr[mi],temp[mi],dens[mi],pres[mi]);
           } 
           fprintf(fp,"\n"); m += n_seg[k];
         }
         k++;
       }
     }
  }
  fclose(fp);
  printf("      %d states extracted\n",state-1);

  free(velr);
  free(velx);
  free(vely);
  free(velz);
  free(temp);
  free(pres);
  free(dens);
  free(xpos);
  free(ypos);
  free(zpos);

  return 0;
}
/*
  NODFOR file
*/
int translate_nodfor(int handle)
{
  int i,j,n,typid,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int *ids, *gids, *groups, *local;
  int nnodes, ngroups, have_local;
  float *xf,*yf,*zf,*e,*xt,*yt,*zt,*et,*xl,*yl,*zl;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/nodfor/metadata") == -1) return 0;
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid < 0) return 0;
  nnodes = length;
  lsda_queryvar(handle,"groups",&typid,&length,&filenum);
  ngroups = length;

  ids   = (int *) malloc(nnodes*sizeof(int));
  local = (int *) malloc(nnodes*sizeof(int));
  groups= (int *) malloc(ngroups*sizeof(int));
  xf    = (float *) malloc(nnodes*sizeof(float));
  yf    = (float *) malloc(nnodes*sizeof(float));
  zf    = (float *) malloc(nnodes*sizeof(float));
  e     = (float *) malloc(nnodes*sizeof(float));
  xt    = (float *) malloc(nnodes*sizeof(float));
  yt    = (float *) malloc(nnodes*sizeof(float));
  zt    = (float *) malloc(nnodes*sizeof(float));
  et    = (float *) malloc(nnodes*sizeof(float));
  lsda_read(handle,LSDA_INT,"ids",0,nnodes,ids);
  lsda_read(handle,LSDA_INT,"local",0,ngroups,local);
  lsda_read(handle,LSDA_INT,"groups",0,ngroups,groups);
  for(i=have_local=0; !have_local && i<ngroups; i++)
    if(local[i] > 0) have_local=1;
  if(have_local) {
    xl = (float *) malloc(ngroups*sizeof(float));
    yl = (float *) malloc(ngroups*sizeof(float));
    zl = (float *) malloc(ngroups*sizeof(float));
  }
  lsda_queryvar(handle,"legend_ids",&typid,&length,&filenum);
  if(typid > 0) {
    gids  = (int *) malloc(ngroups*sizeof(int));
    lsda_read(handle,LSDA_INT,"legend_ids",0,ngroups,gids);
  } else {
    gids=NULL;
  }
/*
  open file and write header
*/
  printf("Extracting NODFOR data\n");
  sprintf(output_file,"%snodfor",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/nodfor/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/nodfor",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",  0,   1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"xforce" ,0, nnodes,xf) !=  nnodes &&
       lsda_read(handle,LSDA_FLOAT,"x_force",0, nnodes,xf) !=  nnodes) break;
    if(lsda_read(handle,LSDA_FLOAT,"yforce" ,0, nnodes,yf) !=  nnodes &&
       lsda_read(handle,LSDA_FLOAT,"y_force",0, nnodes,yf) !=  nnodes) break;
    if(lsda_read(handle,LSDA_FLOAT,"zforce" ,0, nnodes,zf) !=  nnodes &&
       lsda_read(handle,LSDA_FLOAT,"z_force",0, nnodes,zf) !=  nnodes) break;
    if(lsda_read(handle,LSDA_FLOAT,"energy",0, nnodes, e) !=  nnodes) break;
    if(lsda_read(handle,LSDA_FLOAT,"xtotal" ,0, ngroups,xt) !=  ngroups &&
       lsda_read(handle,LSDA_FLOAT,"x_total",0, ngroups,xt) !=  ngroups) break;
    if(lsda_read(handle,LSDA_FLOAT,"ytotal" ,0, ngroups,yt) !=  ngroups &&
       lsda_read(handle,LSDA_FLOAT,"y_total",0, ngroups,yt) !=  ngroups) break;
    if(lsda_read(handle,LSDA_FLOAT,"ztotal" ,0, ngroups,zt) !=  ngroups &&
       lsda_read(handle,LSDA_FLOAT,"z_total",0, ngroups,zt) !=  ngroups) break;
    if(lsda_read(handle,LSDA_FLOAT,"etotal",0,ngroups,et) != ngroups) break;
    if(have_local) {
      if(lsda_read(handle,LSDA_FLOAT,"xlocal" ,0,ngroups,xl) != ngroups &&
         lsda_read(handle,LSDA_FLOAT,"x_local",0,ngroups,xl) != ngroups) break;
      if(lsda_read(handle,LSDA_FLOAT,"ylocal" ,0,ngroups,yl) != ngroups &&
         lsda_read(handle,LSDA_FLOAT,"y_local",0,ngroups,yl) != ngroups) break;
      if(lsda_read(handle,LSDA_FLOAT,"zlocal" ,0,ngroups,zl) != ngroups &&
         lsda_read(handle,LSDA_FLOAT,"z_local",0,ngroups,zl) != ngroups) break;
    }
    fprintf(fp,"\n\n\n\n\n n o d a l   f o r c e   g r o u p");
    fprintf(fp,"    o u t p u t   t=%9.3E\n",time);
    for(i=j=0; i<ngroups; i++) {
      fprintf(fp,"\n\n\n\n\n nodal group output number %2d\n\n",i+1);
      for(n=0; n<groups[i]; n++,j++) {
        fprintf(fp," nd#%8d  xforce=%13.4E   yforce=%13.4E  zforce=%13.4E   energy=%13.4E",
         ids[j],xf[j],yf[j],zf[j],e[j]);
        if(gids) fprintf(fp,"   setid =%8d",gids[i]);
        fprintf(fp,"\n");
      }
      fprintf(fp,"              xtotal=%13.4E   ytotal=%13.4E  ztotal=%13.4E   etotal=%13.4E\n",
         xt[i],yt[i],zt[i],et[i]);
      if(local[i] > 0) {
        fprintf(fp,"              xlocal=%13.4E   ylocal=%13.4E  zlocal=%13.4E\n",
         xl[i],yl[i],zl[i]);
      }
    }
  }
  fclose(fp);
  if(have_local) {
    free(zl);
    free(yl);
    free(xl);
  }
  free(et);
  free(zt);
  free(yt);
  free(xt);
  free(e);
  free(zf);
  free(yf);
  free(xf);
  free(groups);
  free(local);
  free(ids);
  if(gids) free(gids);
  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  BNDOUT file
*/
int translate_bndout(int handle)
{
  int i,k,header,typid,filenum,state;
  LSDA_Length length;
  int have_dn,have_dr,have_p,have_vn,have_vr,have_or;
  FILE *fp;
  BND_DATA dn,dr,p,vn,vr,or;
  float xt,yt,zt,et;
  char title_dir[128];

  lsda_queryvar(handle,"/bndout/discrete/nodes",&typid,&length,&filenum);
  have_dn= (typid >= 0);
  lsda_queryvar(handle,"/bndout/discrete/rigidbodies",&typid,&length,&filenum);
  have_dr= (typid >= 0);
  lsda_queryvar(handle,"/bndout/pressure",&typid,&length,&filenum);
  have_p= (typid >= 0);
  lsda_queryvar(handle,"/bndout/velocity/nodes",&typid,&length,&filenum);
  have_vn= (typid >= 0);
  lsda_queryvar(handle,"/bndout/velocity/rigidbodies",&typid,&length,&filenum);
  have_vr= (typid >= 0);
  lsda_queryvar(handle,"/bndout/orientation/rigidbodies",&typid,&length,&filenum);
  have_or= (typid >= 0);
/*
  Read metadata
*/
  title_dir[0]=0;
  if(have_dn) {
    lsda_cd(handle,"/bndout/discrete/nodes/metadata");
    strcpy(title_dir,"/bndout/discrete/nodes/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    dn.num = length;
    dn.ids = (int *) malloc(dn.num*sizeof(int));
    dn.xf = (float *) malloc(4*dn.num*sizeof(float));
    dn.yf = dn.xf + dn.num;
    dn.zf = dn.yf + dn.num;
    dn.e  = dn.zf + dn.num;
    lsda_read(handle,LSDA_INT,"ids",0,dn.num,dn.ids);
  }
  if(have_dr) {
    lsda_cd(handle,"/bndout/discrete/rigidbodies/metadata");
    strcpy(title_dir,"/bndout/discrete/rigidbodies/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    dr.num = length;
    dr.ids = (int *) malloc(dr.num*sizeof(int));
    dr.xf = (float *) malloc(4*dr.num*sizeof(float));
    dr.yf = dr.xf + dr.num;
    dr.zf = dr.yf + dr.num;
    dr.e  = dr.zf + dr.num;
    lsda_read(handle,LSDA_INT,"ids",0,dr.num,dr.ids);
  }
  if(have_p) {
    lsda_cd(handle,"/bndout/pressure/metadata");
    strcpy(title_dir,"/bndout/pressure/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    p.num = length;
    p.ids = (int *) malloc(p.num*sizeof(int));
    p.xf = (float *) malloc(4*p.num*sizeof(float));
    p.yf = p.xf + p.num;
    p.zf = p.yf + p.num;
    p.e  = p.zf + p.num;
    lsda_read(handle,LSDA_INT,"ids",0,p.num,p.ids);
  }
  if(have_vn) {
    lsda_cd(handle,"/bndout/velocity/nodes/metadata");
    strcpy(title_dir,"/bndout/velocity/nodes/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    vn.num = length;
    vn.ids = (int *) malloc(vn.num*sizeof(int));
    vn.setid = (int *) malloc(vn.num*sizeof(int));
    vn.xf = (float *) malloc(4*vn.num*sizeof(float));
    vn.yf = vn.xf + vn.num;
    vn.zf = vn.yf + vn.num;
    vn.e  = vn.zf + vn.num;
    lsda_read(handle,LSDA_INT,"ids",0,vn.num,vn.ids);
    lsda_read(handle,LSDA_INT,"setid",0,vn.num,vn.setid);
  }
  if(have_vr) {
    lsda_cd(handle,"/bndout/velocity/rigidbodies/metadata");
    strcpy(title_dir,"/bndout/velocity/rigidbodies/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    vr.num = length;
    vr.ids = (int *) malloc(vr.num*sizeof(int));
    vr.xf = (float *) malloc(7*vr.num*sizeof(float));
    vr.yf = vr.xf + vr.num;
    vr.zf = vr.yf + vr.num;
    vr.e  = vr.zf + vr.num;
    vr.xm = vr.e  + vr.num;
    vr.ym = vr.xm + vr.num;
    vr.zm = vr.ym + vr.num;
    lsda_read(handle,LSDA_INT,"ids",0,vr.num,vr.ids);
  }
  if(have_or) {
    lsda_cd(handle,"/bndout/orientation/rigidbodies/metadata");
    strcpy(title_dir,"/bndout/orientation/rigidbodies/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    or.num = length;
    or.ids = (int *) malloc(or.num*sizeof(int));
    or.xf = (float *) malloc(7*or.num*sizeof(float));
    or.yf = or.xf + or.num;
    or.zf = or.yf + or.num;
    or.e  = or.zf + or.num;
    or.xm = or.e  + or.num;
    or.ym = or.xm + or.num;
    or.zm = or.ym + or.num;
    lsda_read(handle,LSDA_INT,"ids",0,or.num,or.ids);
  }

  if(strlen(title_dir) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  printf("Extracting BNDOUT data\n");
  sprintf(output_file,"%sbndout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_dir,fp);
  k=0;
  if(have_vn) {
    lsda_cd(handle,"/bndout/velocity/nodes/metadata");
    i = have_vr ? 0 : 1;
    k = 1;
    output_legend(handle,fp,1,i);
  }
  if(have_vr) {
    lsda_cd(handle,"/bndout/velocity/rigidbodies/metadata");
    i = !k;
    output_legend(handle,fp,i,1);
  }
  if(have_or) {
    lsda_cd(handle,"/bndout/orientation/rigidbodies/metadata");
    i = !k;
    output_legend(handle,fp,i,1);
  }
/*
  Loop through time states and write each one
*/
  for(state=1; ; state++) {
    header=0;
    xt = yt = zt = et = 0.;
    if(have_dn && ! bndout_dn(fp,handle,state,&dn,&xt,&yt,&zt,&et,&header)) break;
    if(have_dr && ! bndout_dr(fp,handle,state,&dr,&xt,&yt,&zt,&et,&header)) break;
    if(have_dn || have_dr) {
      fprintf(fp,"              xtotal=%13.4E   ytotal=%13.4E  ztotal=%13.4E   etotal=%13.4E\n",
      xt,yt,zt,et);
    }
    if(have_p && ! bndout_p(fp,handle,state,&p,&header)) break;
    xt = yt = zt = et = 0.;
    if(have_vn && ! bndout_vn(fp,handle,state,&vn,&xt,&yt,&zt,&et,&header)) break;
    if(have_vr && ! bndout_vr(fp,handle,state,&vr,&xt,&yt,&zt,&et,&header)) break;
    if(have_vn || have_vr) {
      fprintf(fp,"              xtotal=%13.4E   ytotal=%13.4E  ztotal=%13.4E   etotal=%13.4E\n",
      xt,yt,zt,et);
    if(have_or && ! bndout_or(fp,handle,state,&or,&xt,&yt,&zt,&et,&header)) break;
    }
  }
  fclose(fp);
/*
  free everything here....
*/
  if(have_dn) {
    free(dn.ids);
    free(dn.xf);
  }
  if(have_dr) {
    free(dr.ids);
    free(dr.xf);
  }
  if(have_p) {
    free(p.ids);
    free(p.xf);
  }
  if(have_vn) {
    free(vn.ids);
    free(vn.xf);
    free(vn.setid);
  }
  if(have_vr) {
    free(vr.ids);
    free(vr.xf);
  }
  printf("      %d states extracted\n",state-1);
  return 0;
}


int
bndout_dn(FILE *fp,int handle, int state, BND_DATA *dn,
          float *xt, float *yt, float *zt, float *et, int *header)
{
  char dirname[128];
  float time;
  int typid, filenum;
  LSDA_Length length;
  int i;

  if(state<=999999)
    sprintf(dirname,"/bndout/discrete/nodes/d%6.6d",state);
  else
    sprintf(dirname,"/bndout/discrete/nodes/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xforce" ,0,dn->num,dn->xf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"x_force",0,dn->num,dn->xf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yforce" ,0,dn->num,dn->yf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"y_force",0,dn->num,dn->yf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zforce" ,0,dn->num,dn->zf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"z_force",0,dn->num,dn->zf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"energy",0,dn->num,dn->e)  != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xtotal" ,0,1,xt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"x_total",0,1,xt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ytotal" ,0,1,yt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"y_total",0,1,yt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ztotal" ,0,1,zt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"z_total",0,1,zt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"etotal",0,1,et)  != 1) return 0;
/*
  Output data
*/
  fprintf(fp,"\n\n\n\n\n n o d a l   f o r c e/e n e r g y");
  fprintf(fp,"    o u t p u t   t=%9.3E\n",time);
  *header = 1;
  fprintf(fp,"\n\n\n\n\n discrete nodal point forces \n\n");
  for(i=0; i<dn->num; i++) {
    fprintf(fp," nd#%8d  xforce=%13.4E   yforce=%13.4E  zforce=%13.4E   energy=%13.4E\n",
      dn->ids[i],dn->xf[i],dn->yf[i],dn->zf[i],dn->e[i]);
  }
  return 1;
}

int
bndout_dr(FILE *fp,int handle, int state, BND_DATA *dn,
          float *xtt, float *ytt, float *ztt, float *ett, int *header)
{
  char dirname[128];
  float time;
  int typid, filenum;
  LSDA_Length length;
  float xt,yt,zt,et;
  int i;

  if(state<=999999)
    sprintf(dirname,"/bndout/discrete/rigidbodies/d%6.6d",state);
  else
    sprintf(dirname,"/bndout/discrete/rigidbodies/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xforce" ,0,dn->num,dn->xf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"x_force",0,dn->num,dn->xf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yforce" ,0,dn->num,dn->yf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"y_force",0,dn->num,dn->yf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zforce" ,0,dn->num,dn->zf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"z_force",0,dn->num,dn->zf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"energy",0,dn->num,dn->e)  != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xtotal" ,0,1,&xt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"x_total",0,1,&xt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ytotal" ,0,1,&yt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"y_total",0,1,&yt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ztotal" ,0,1,&zt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"z_total",0,1,&zt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"etotal",0,1,&et)  != 1) return 0;
  *xtt += xt;
  *ytt += yt;
  *ztt += zt;
  *ett += et;
/*
  Output data
*/
  if(! *header) {
    fprintf(fp,"\n\n\n\n\n n o d a l   f o r c e/e n e r g y");
    fprintf(fp,"    o u t p u t   t=%9.3E\n",time);
    fprintf(fp,"\n\n\n\n\n discrete nodal point forces \n\n");
  }
  *header = 2;
  for(i=0; i<dn->num; i++) {
    fprintf(fp," mt#%8d  xforce=%13.4E   yforce=%13.4E  zforce=%13.4E   energy=%13.4E\n",
      dn->ids[i],dn->xf[i],dn->yf[i],dn->zf[i],dn->e[i]);
  }
  return 1;
}

int
bndout_p(FILE *fp,int handle, int state, BND_DATA *dn, int *header)
{
  char dirname[128];
  float time;
  int typid, filenum;
  LSDA_Length length;
  float xt,yt,zt,et;
  int i;

  if(state<=999999)
    sprintf(dirname,"/bndout/pressure/d%6.6d",state);
  else
    sprintf(dirname,"/bndout/pressure/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xforce" ,0,dn->num,dn->xf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"x_force",0,dn->num,dn->xf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yforce" ,0,dn->num,dn->yf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"y_force",0,dn->num,dn->yf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zforce" ,0,dn->num,dn->zf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"z_force",0,dn->num,dn->zf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"energy",0,dn->num,dn->e)  != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xtotal" ,0,1,&xt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"x_total",0,1,&xt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ytotal" ,0,1,&yt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"y_total",0,1,&yt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ztotal" ,0,1,&zt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"z_total",0,1,&zt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"etotal",0,1,&et)  != 1) return 0;
/*
  Output data
*/
  fprintf(fp,"\n\n\n\n\n n o d a l   f o r c e/e n e r g y");
  fprintf(fp,"    o u t p u t   t=%9.3E\n",time);
  *header = 3;
  fprintf(fp,"\n\n\n\n\n pressure boundary condition forces \n\n");
  for(i=0; i<dn->num; i++) {
    fprintf(fp," nd#%8d  xforce=%13.4E   yforce=%13.4E  zforce=%13.4E   energy=%13.4E\n",
      dn->ids[i],dn->xf[i],dn->yf[i],dn->zf[i],dn->e[i]);
  }
  fprintf(fp,"              xtotal=%13.4E   ytotal=%13.4E  ztotal=%13.4E   etotal=%13.4E\n",
  xt,yt,zt,et);
  return 1;
}

int
bndout_vn(FILE *fp,int handle, int state, BND_DATA *dn,
          float *xt, float *yt, float *zt, float *et, int *header)
{
  char dirname[128];
  float time;
  int typid, filenum;
  LSDA_Length length;
  int i;

  if(state<=999999)
    sprintf(dirname,"/bndout/velocity/nodes/d%6.6d",state);
  else
    sprintf(dirname,"/bndout/velocity/nodes/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xforce" ,0,dn->num,dn->xf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"x_force",0,dn->num,dn->xf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yforce" ,0,dn->num,dn->yf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"y_force",0,dn->num,dn->yf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zforce" ,0,dn->num,dn->zf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"z_force",0,dn->num,dn->zf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"energy",0,dn->num,dn->e)  != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xtotal" ,0,1,xt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"x_total",0,1,xt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ytotal" ,0,1,yt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"y_total",0,1,yt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ztotal" ,0,1,zt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"z_total",0,1,zt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"etotal",0,1,et)  != 1) return 0;
/*
  Output data
*/
  fprintf(fp,"\n\n\n\n\n n o d a l   f o r c e/e n e r g y");
  fprintf(fp,"    o u t p u t   t=%9.3E\n",time);
  *header = 4;
  fprintf(fp,"\n\n\n\n\n velocity boundary condition forces/");
  fprintf(fp,"rigid body moments \n\n");
  for(i=0; i<dn->num; i++) {
    fprintf(fp," nd#%8d  xforce=%13.4E   yforce=%13.4E  zforce=%13.4E   energy=%13.4E   setid =%8d\n",
      dn->ids[i],dn->xf[i],dn->yf[i],dn->zf[i],dn->e[i],dn->setid[i]);
  }
  return 1;
}

int
bndout_vr(FILE *fp,int handle, int state, BND_DATA *dn,
          float *xtt, float *ytt, float *ztt, float *ett, int *header)
{
  char dirname[128];
  float time;
  int typid, filenum;
  LSDA_Length length;
  float xt,yt,zt,et;
  int i;

  if(state<=999999)
    sprintf(dirname,"/bndout/velocity/rigidbodies/d%6.6d",state);
  else
    sprintf(dirname,"/bndout/velocity/rigidbodies/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xforce" ,0,dn->num,dn->xf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"x_force",0,dn->num,dn->xf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yforce" ,0,dn->num,dn->yf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"y_force",0,dn->num,dn->yf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zforce" ,0,dn->num,dn->zf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"z_force",0,dn->num,dn->zf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"energy",0,dn->num,dn->e)  != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xmoment",0,dn->num,dn->xm) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ymoment",0,dn->num,dn->ym) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zmoment",0,dn->num,dn->zm) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xtotal" ,0,1,&xt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"x_total",0,1,&xt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ytotal" ,0,1,&yt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"y_total",0,1,&yt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ztotal" ,0,1,&zt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"z_total",0,1,&zt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"etotal",0,1,&et)  != 1) return 0;
  *xtt += xt;
  *ytt += yt;
  *ztt += zt;
  *ett += et;
/*
  Output data
*/
  if(*header != 4) {
    fprintf(fp,"\n\n\n\n\n n o d a l   f o r c e/e n e r g y");
    fprintf(fp,"    o u t p u t   t=%9.3E\n",time);
    fprintf(fp,"\n\n\n\n\n velocity boundary condition forces/");
    fprintf(fp,"rigid body moments \n\n");
  }
  *header = 5;
  for(i=0; i<dn->num; i++) {
    fprintf(fp,"mat#%8d  xforce=%13.4E   yforce=%13.4E  zforce=%13.4E   energy=%13.4E\n",
      dn->ids[i],dn->xf[i],dn->yf[i],dn->zf[i],dn->e[i]);
    fprintf(fp,"             xmoment=%13.4E  ymoment=%13.4E zmoment=%13.4E\n",
      dn->xm[i],dn->ym[i],dn->zm[i]);
  }
  return 1;
}
/*
  BNDOUT - prescribed orientation start
*/
int
bndout_or(FILE *fp,int handle, int state, BND_DATA *dn,
          float *xtt, float *ytt, float *ztt, float *ett, int *header)
{
  char dirname[128];
  float time;
  int typid, filenum;
  LSDA_Length length;
  float xt,yt,zt,et;
  int i;

  if(state<=999999)
    sprintf(dirname,"/bndout/orientation/rigidbodies/d%6.6d",state);
  else
    sprintf(dirname,"/bndout/orientation/rigidbodies/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xforce" ,0,dn->num,dn->xf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"x_force",0,dn->num,dn->xf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"yforce" ,0,dn->num,dn->yf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"y_force",0,dn->num,dn->yf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zforce" ,0,dn->num,dn->zf) != dn->num &&
     lsda_read(handle,LSDA_FLOAT,"z_force",0,dn->num,dn->zf) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"energy",0,dn->num,dn->e)  != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xmoment",0,dn->num,dn->xm) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ymoment",0,dn->num,dn->ym) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"zmoment",0,dn->num,dn->zm) != dn->num) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"xtotal" ,0,1,&xt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"x_total",0,1,&xt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ytotal" ,0,1,&yt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"y_total",0,1,&yt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"ztotal" ,0,1,&zt)  != 1 &&
     lsda_read(handle,LSDA_FLOAT,"z_total",0,1,&zt)  != 1) return 0;
  if(lsda_read(handle,LSDA_FLOAT,"etotal",0,1,&et)  != 1) return 0;
  *xtt += xt;
  *ytt += yt;
  *ztt += zt;
  *ett += et;
/*
  Output data
*/
  if(*header != 4) {
    fprintf(fp,"\n\n\n\n\n n o d a l   f o r c e/e n e r g y");
    fprintf(fp,"    o u t p u t   t=%9.3E\n",time);
    fprintf(fp,"\n\n\n\n\n orientation boundary condition rigid body moments \n\n");
  }
  *header = 5;
  for(i=0; i<dn->num; i++) {
    fprintf(fp,"mat#%8d  xmoment=%13.4E   ymoment=%13.4E  zmoment=%13.4E   energy=%13.4E\n",
      dn->ids[i],dn->xm[i],dn->ym[i],dn->zm[i],dn->e[i]);
  }
  return 1;
}
/*
  BNDOUT - prescribed orientation end
*/
/*
  RBDOUT file
*/
int translate_rbdout(int handle)
{
  int i,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int num_nodal;
  int max_num;
  int *ids,cycle;
  float *gx,*gy,*gz;
  float *gdx,*gdy,*gdz,*grdx,*grdy,*grdz;
  float *gvx,*gvy,*gvz,*grvx,*grvy,*grvz;
  float *gax,*gay,*gaz,*grax,*gray,*graz;
  float *ldx,*ldy,*ldz,*lrdx,*lrdy,*lrdz;
  float *lvx,*lvy,*lvz,*lrvx,*lrvy,*lrvz;
  float *lax,*lay,*laz,*lrax,*lray,*lraz;
  float *d11,*d12,*d13,*d21,*d22,*d23,*d31,*d32,*d33;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/rbdout/metadata") == -1) return 0;
  printf("Extracting RBDOUT data\n");
/*
  Read metadata

  In older rbdout files, there is an id vector and a "num_nodal"
  value in the metadata directory, and the number of entries in each
  state directory is the same.  This was found to be a problem when
  rigid deformable material switching occurs, so now there is an
  id array and num_nodal value in each directory.  Of course, we
  have to support both cases....
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  if(typid > 0) {
    max_num = num = length;
    ids = (int *) malloc(num*sizeof(int));
    lsda_read(handle,LSDA_INT,"ids",0,num,ids);
    lsda_read(handle,LSDA_INT,"num_nodal",0,1,&num_nodal);
  } else {
    max_num = num = 100;
    ids = (int *) malloc(num*sizeof(int));
  }
  gx  = (float *) malloc(48*num*sizeof(float));
  gy  = gx   + num;
  gz  = gy   + num;
  gdx = gz   + num;
  gdy = gdx  + num;
  gdz = gdy  + num;
  grdx= gdz  + num;
  grdy= grdx + num;
  grdz= grdy + num;
  gvx = grdz + num;
  gvy = gvx  + num;
  gvz = gvy  + num;
  grvx= gvz  + num;
  grvy= grvx + num;
  grvz= grvy + num;
  gax = grvz + num;
  gay = gax  + num;
  gaz = gay  + num;
  grax= gaz  + num;
  gray= grax + num;
  graz= gray + num;
  ldx = graz + num;
  ldy = ldx  + num;
  ldz = ldy  + num;
  lrdx= ldz  + num;
  lrdy= lrdx + num;
  lrdz= lrdy + num;
  lvx = lrdz + num;
  lvy = lvx  + num;
  lvz = lvy  + num;
  lrvx= lvz  + num;
  lrvy= lrvx + num;
  lrvz= lrvy + num;
  lax = lrvz + num;
  lay = lax  + num;
  laz = lay  + num;
  lrax= laz  + num;
  lray= lrax + num;
  lraz= lray + num;
  d11 = lraz + num;
  d12 = d11  + num;
  d13 = d12  + num;
  d21 = d13  + num;
  d22 = d21  + num;
  d23 = d22  + num;
  d31 = d23  + num;
  d32 = d31  + num;
  d33 = d32  + num;

/*
  open file and write header
*/
  sprintf(output_file,"%srbdout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/rbdout/metadata",fp);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/rbdout",dp,dirname)) != NULL; state++) {
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    if(typid > 0) {
      num = length;
      if(num > max_num) {   /* have to reallocate arrays */
        max_num = num+10;
        free(ids);
        free(gx);
        ids = (int *) malloc(max_num*sizeof(int));
        gx  = (float *) malloc(48*max_num*sizeof(float));
        gy  = gx   + max_num;
        gz  = gy   + max_num;
        gdx = gz   + max_num;
        gdy = gdx  + max_num;
        gdz = gdy  + max_num;
        grdx= gdz  + max_num;
        grdy= grdx + max_num;
        grdz= grdy + max_num;
        gvx = grdz + max_num;
        gvy = gvx  + max_num;
        gvz = gvy  + max_num;
        grvx= gvz  + max_num;
        grvy= grvx + max_num;
        grvz= grvy + max_num;
        gax = grvz + max_num;
        gay = gax  + max_num;
        gaz = gay  + max_num;
        grax= gaz  + max_num;
        gray= grax + max_num;
        graz= gray + max_num;
        ldx = graz + max_num;
        ldy = ldx  + max_num;
        ldz = ldy  + max_num;
        lrdx= ldz  + max_num;
        lrdy= lrdx + max_num;
        lrdz= lrdy + max_num;
        lvx = lrdz + max_num;
        lvy = lvx  + max_num;
        lvz = lvy  + max_num;
        lrvx= lvz  + max_num;
        lrvy= lrvx + max_num;
        lrvz= lrvy + max_num;
        lax = lrvz + max_num;
        lay = lax  + max_num;
        laz = lay  + max_num;
        lrax= laz  + max_num;
        lray= lrax + max_num;
        lraz= lray + max_num;
        d11 = lraz + max_num;
        d12 = d11  + max_num;
        d13 = d12  + max_num;
        d21 = d13  + max_num;
        d22 = d21  + max_num;
        d23 = d22  + max_num;
        d31 = d23  + max_num;
        d32 = d31  + max_num;
        d33 = d32  + max_num;
      }
      lsda_read(handle,LSDA_INT,"ids",0,num,ids);
      lsda_read(handle,LSDA_INT,"num_nodal",0,1,&num_nodal);
    }
    if(num==0) continue;  /* don't output a header if there is no data */
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_x",0,num,gx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_y",0,num,gy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_z",0,num,gz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_dx",0,num,gdx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_dy",0,num,gdy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_dz",0,num,gdz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_rdx",0,num,grdx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_rdy",0,num,grdy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_rdz",0,num,grdz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_vx",0,num,gvx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_vy",0,num,gvy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_vz",0,num,gvz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_rvx",0,num,grvx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_rvy",0,num,grvy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_rvz",0,num,grvz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_ax",0,num,gax) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_ay",0,num,gay) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_az",0,num,gaz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_rax",0,num,grax) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_ray",0,num,gray) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"global_raz",0,num,graz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_11",0,num,d11) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_12",0,num,d12) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_13",0,num,d13) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_21",0,num,d21) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_22",0,num,d22) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_23",0,num,d23) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_31",0,num,d31) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_32",0,num,d32) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"dircos_33",0,num,d33) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_dx",0,num,ldx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_dy",0,num,ldy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_dz",0,num,ldz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_rdx",0,num,lrdx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_rdy",0,num,lrdy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_rdz",0,num,lrdz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_vx",0,num,lvx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_vy",0,num,lvy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_vz",0,num,lvz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_rvx",0,num,lrvx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_rvy",0,num,lrvy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_rvz",0,num,lrvz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_ax",0,num,lax) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_ay",0,num,lay) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_az",0,num,laz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_rax",0,num,lrax) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_ray",0,num,lray) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"local_raz",0,num,lraz) != num) break;

    fprintf(fp,"1\n\n  r i g i d   b o d y   m o t i o n   a t  cycle=%8d    time=%15.6E\n",cycle,time);
    for(i=0; i<num; i++) {
      if(i < num-num_nodal)
        fprintf(fp,"\n rigid body%8d\n",ids[i]);
      else
        fprintf(fp,"\n nodal rigid body%8d\n",ids[i]);
      fprintf(fp,"   global             x           y           z");
      fprintf(fp,"          x-rot       y-rot       z-rot\n");
      fprintf(fp,"   coordinates:  %12.4E%12.4E%12.4E\n",gx[i],gy[i],gz[i]);
      fprintf(fp," displacements:  %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
        gdx[i],gdy[i],gdz[i],grdx[i],grdy[i],grdz[i]);
      fprintf(fp,"    velocities:  %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
        gvx[i],gvy[i],gvz[i],grvx[i],grvy[i],grvz[i]);
      fprintf(fp," accelerations:  %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
        gax[i],gay[i],gaz[i],grax[i],gray[i],graz[i]);
      fprintf(fp,"\nprincipal or user defined local coordinate direction vectors\n");
      fprintf(fp,"                     a           b           c\n");
      fprintf(fp,"     row 1%15.4E%15.4E%15.4E\n",d11[i],d12[i],d13[i]);
      fprintf(fp,"     row 2%15.4E%15.4E%15.4E\n",d21[i],d22[i],d23[i]);
      fprintf(fp,"     row 3%15.4E%15.4E%15.4E\n\n",d31[i],d32[i],d33[i]);
      fprintf(fp," output in principal or user defined local coordinate directions\n");
      fprintf(fp,"                      a           b           c          ");
      fprintf(fp,"a-rot       b-rot       c-rot\n");
      fprintf(fp," displacements:  %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
        ldx[i],ldy[i],ldz[i],lrdx[i],lrdy[i],lrdz[i]);
      fprintf(fp,"    velocities:  %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
        lvx[i],lvy[i],lvz[i],lrvx[i],lrvy[i],lrvz[i]);
      fprintf(fp," accelerations:  %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
        lax[i],lay[i],laz[i],lrax[i],lray[i],lraz[i]);
    }

  }
  fclose(fp);
  free(gx);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  GCEOUT file
*/
int translate_gceout(int handle)
{
  int i,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int *ids;
  float *xf,*yf,*zf,*xm,*ym,*zm,*tf,*tm;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/gceout/metadata") == -1) return 0;
  printf("Extracting GCEOUT data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  ids = (int *) malloc(num*sizeof(int));
  xf  = (float *) malloc(8*num*sizeof(float));
  yf  = xf   + num;
  zf  = yf   + num;
  xm  = zf   + num;
  ym  = xm   + num;
  zm  = ym   + num;
  tf  = zm   + num;
  tm  = tf   + num;

  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%sgceout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/gceout/metadata",fp);
  fprintf(fp,"\n\n\n\n c o n t a c t   e n t i t i e s   r e s u l t");
  fprintf(fp," a n t s\n\n\n\n");
  fprintf(fp,"       material#   time      x-force     y-force     z-force");
  fprintf(fp,"    magnitude\n");
  fprintf(fp,"line#2  resultant moments    x-moment    y-moment    z-moment");
  fprintf(fp,"   magnitude\n");
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/gceout",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_force",         0,num,xf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_force",         0,num,yf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_force",         0,num,zf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_moment",        0,num,xm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_moment",        0,num,ym) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_moment",        0,num,zm) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"force_magnitude", 0,num,tf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"moment_magnitude",0,num,tm) != num) break;
    for(i=0; i<num; i++) {
      fprintf(fp,"%12d  %12.4E%12.4E%12.4E%12.4E%12.4E\n",
       ids[i],time,xf[i],yf[i],zf[i],tf[i]);
      fprintf(fp,"                          %12.4E%12.4E%12.4E%12.4E\n",
       xm[i],ym[i],zm[i],tm[i]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
  free(xf);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  SLEOUT file
*/
int translate_sleout(int handle)
{
  int i,j,typid,num,numm,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int *ids,*single_sided, have_friction = -1;
  int *is_transducer;
  float *slave, *master, *friction;
  float time,ts,tm,te,tf;
  int cycle;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/sleout/metadata") == -1) return 0;
  printf("Extracting SLEOUT data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  ids          = (int *) malloc(num*sizeof(int));
  single_sided = (int *) malloc(num*sizeof(int));
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
  lsda_read(handle,LSDA_INT,"single_sided",0,num,single_sided);
  lsda_queryvar(handle,"is_transducer",&typid,&length,&filenum);
  if(length == num) {
    is_transducer= (int *) malloc(num*sizeof(int));
    lsda_read(handle,LSDA_INT,"is_transducer",0,num,is_transducer);
  } else {
    is_transducer= NULL;
  }
  numm = num;
  for(i=0; i<num; i++)
    if(single_sided[i]) numm--;
  friction  = (float *) malloc(num *sizeof(float));
  slave  = (float *) malloc(num *sizeof(float));
  master = (float *) malloc(numm*sizeof(float));
/*
  open file and write header
*/
  sprintf(output_file,"%ssleout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/sleout/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/sleout",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"slave",0,num,slave) != num) break;
    if(numm > 0 &&
       lsda_read(handle,LSDA_FLOAT,"master",0,numm,master) != numm) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_slave",0,1,&ts) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_master",0,1,&tm) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"total_energy",0,1,&te) != 1) break;
    if(have_friction == -1) {  /* haven't checked yet... */
      lsda_queryvar(handle,"total_friction",&typid,&length,&filenum);
      if(length > 0)
        have_friction=1;
      else
        have_friction=0;
    }
    if(have_friction) {
      lsda_read(handle,LSDA_FLOAT,"total_friction",0,1,&tf);
      lsda_read(handle,LSDA_FLOAT,"friction_energy",0,num,friction);
      fprintf(fp,"\n\n\n contact interface energy file, time=%14.7E\n",time);
      fprintf(fp,"    #            slave       master        cycle=%14d",cycle);
      if(is_transducer) {
        fprintf(fp,"    frictional  transducer\n\n");
      } else {
        fprintf(fp,"    frictional\n\n");
      }
      for(i=j=0; i<num; i++) {
        if(single_sided[i])
          fprintf(fp,"%10d%13.4E%13.4E",ids[i],slave[i],0.0);
        else
          fprintf(fp,"%10d%13.4E%13.4E",ids[i],slave[i],master[j++]);
        fprintf(fp,"                            %13.4E",friction[i]);
        if(is_transducer)
          fprintf(fp,"%8d",is_transducer[i]);
        fprintf(fp,"\n");
      }
      fprintf(fp,"\n summary   total slave side=%14.7E\n",ts);
      fprintf(fp,  "           total mastr side=%14.7E\n",tm);
      fprintf(fp,  "           total energy    =%14.7E\n",te);
      fprintf(fp,  "           friction energy =%14.7E\n\n\n\n",tf);
    } else {
      fprintf(fp,"\n\n\n contact interface energy file, time=%14.7E\n",time);
      fprintf(fp,"    #            slave       master        cycle=%14d",cycle);
      if(is_transducer) {
        fprintf(fp,"                transducer\n\n");
      } else {
        fprintf(fp,"\n\n");
      }
      for(i=j=0; i<num; i++) {
        if(single_sided[i])
          fprintf(fp,"%10d%13.4E%13.4E",ids[i],slave[i],0.0);
        else
          fprintf(fp,"%10d%13.4E%13.4E",ids[i],slave[i],master[j++]);
        if(is_transducer)
          fprintf(fp,"                                         %8d",is_transducer[i]);
        fprintf(fp,"\n");
      }
      fprintf(fp,"\n summary   total slave side=%14.7E\n",ts);
      fprintf(fp,  "           total mastr side=%14.7E\n",tm);
      fprintf(fp,  "           total energy    =%14.7E\n\n\n\n",te);
    }
  }
  fclose(fp);
  free(master);
  free(slave);
  free(friction);
  if(is_transducer) free(is_transducer);
  free(single_sided);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  SBTOUT file
*/
int translate_sbtout(int handle)
{
  int i,typid,filenum,state;
  int numb,nums,numr;
  LSDA_Length length;
  char dirname[256];
  int *bids,*sids,*rids;
  float *blength,*bforce,*slip,*pullout,*rforce;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/sbtout/metadata") == -1) return 0;
  printf("Extracting SBTOUT data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"belt_ids",&typid,&length,&filenum);
  numb = length;
  bids  = (int *) malloc(numb*sizeof(int));
  blength = (float *) malloc(numb*sizeof(float));
  bforce = (float *) malloc(numb*sizeof(float));
  lsda_read(handle,LSDA_INT,"belt_ids",0,numb,bids);
  lsda_queryvar(handle,"slipring_ids",&typid,&length,&filenum);
  nums = length;
  if(typid > 0 && nums > 0) {
    sids  = (int *) malloc(nums*sizeof(int));
    slip  = (float *) malloc(nums*sizeof(float));
    lsda_read(handle,LSDA_INT,"slipring_ids",0,nums,sids);
  } else {
    nums=0;
  }
  lsda_queryvar(handle,"retractor_ids",&typid,&length,&filenum);
  numr = length;
  if(typid > 0 && numr > 0) {
    rids  = (int *) malloc(numr*sizeof(int));
    pullout  = (float *) malloc(numr*sizeof(float));
    rforce  = (float *) malloc(numr*sizeof(float));
    lsda_read(handle,LSDA_INT,"retractor_ids",0,numr,rids);
  } else {
    numr=0;
  }
/*
  open file and write header
*/
  sprintf(output_file,"%ssbtout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/sbtout/metadata",fp);
  output_legend(handle,fp,1,1);
  fprintf(fp,"\n\n\n\n S E A T B E L T    O U T P U T\n\n");
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/sbtout",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(numb && lsda_read(handle,LSDA_FLOAT,"belt_force",0,numb,bforce) != numb) break;
    if(numb && lsda_read(handle,LSDA_FLOAT,"belt_length",0,numb,blength) != numb) break;
    if(nums && lsda_read(handle,LSDA_FLOAT,"ring_slip",0,nums,slip) != nums) break;
    if(numr) {
      if(lsda_read(handle,LSDA_FLOAT,"retractor_pull_out",0,numr,pullout) != numr) break;
      if(lsda_read(handle,LSDA_FLOAT,"retractor_force",0,numr,rforce) != numr) break;
    }

    fprintf(fp,"\n\n time...........................%16.5E\n",time);
    for(i=0; i<numb; i++) {
      fprintf(fp,"\n seat belt number...............%8d\n",bids[i]);
      fprintf(fp,  " force..........................%16.5E\n",bforce[i]);
      fprintf(fp,  " current length.................%16.5E\n",blength[i]);
    }
    for(i=0; i<nums; i++) {
      fprintf(fp,"\n slip ring number...............%8d\n",sids[i]);
      fprintf(fp,  " total slip from side 1 to .....%16.5E\n",slip[i]);
    }
    for(i=0; i<numr; i++) {
      fprintf(fp,"\n retractor number...............%8d\n",rids[i]);
      fprintf(fp,  " pull-out to date...............%14.5E\n",pullout[i]);
      fprintf(fp,  " force in attached element......%14.5E\n",rforce[i]);
    }
  }
  fclose(fp);
  if(numr > 0) {
    free(rforce);
    free(pullout);
    free(rids);
  }
  if(nums > 0) {
    free(slip);
    free(sids);
  }
  free(bforce);
  free(blength);
  free(bids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  JNTFORC file
*/
int translate_jntforc(int handle)
{
  int i,j,k,typid,filenum,state;
  int numj,num0,num1;
  LSDA_Length length;
  char dirname[256];
  int *idj,*local,*id0,*id1;
  int have_type1_extra = -1;
  float *xf,*yf,*zf,*xm,*ym,*zm,*rf,*rm;
  float *p0,*dpdt0,*t0,*dtdt0,*s0,*dsdt0;
  float *ps0,*pd0,*pt0,*ts0,*td0,*tt0,*ss0,*sd0,*st0,*je0;
  float *a1,*dadt1,*g1,*dgdt1,*b1,*dbdt1;
  float *as1,*ad1,*at1,*gsf1,*bs1,*bd1,*bt1,*je1,*jx1,*jx2;
  float *e;
  float time;
  FILE *fp;
  char title_dir[128];

  printf("Extracting JNTFORC data\n");
/*
  Read metadata
*/
  title_dir[0]=0;
  lsda_queryvar(handle,"/jntforc/joints",&typid,&length,&filenum);
  e=NULL;
  if(typid == 0) {
    lsda_cd(handle,"/jntforc/joints/metadata");
    strcpy(title_dir,"/jntforc/joints/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    numj = length;
    idj  = (int *) malloc(numj*sizeof(int));
    local  = (int *) malloc(numj*sizeof(int));
    xf  = (float *) malloc(numj*sizeof(float));
    yf  = (float *) malloc(numj*sizeof(float));
    zf  = (float *) malloc(numj*sizeof(float));
    xm  = (float *) malloc(numj*sizeof(float));
    ym  = (float *) malloc(numj*sizeof(float));
    zm  = (float *) malloc(numj*sizeof(float));
    rf  = (float *) malloc(numj*sizeof(float));
    rm  = (float *) malloc(numj*sizeof(float));
    lsda_read(handle,LSDA_INT,"ids",0,numj,idj);
    lsda_read(handle,LSDA_INT,"local",0,numj,local);
    lsda_query(handle,"/jntforc/joints/d000001/energy",&typid,&length);
    if(length==numj) e  = (float *) malloc(numj*sizeof(float));
  } else {
    numj = 0;
  }
  lsda_queryvar(handle,"/jntforc/type0",&typid,&length,&filenum);
  if(typid == 0) {
    lsda_cd(handle,"/jntforc/type0/metadata");
    strcpy(title_dir,"/jntforc/type0/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    num0 = length;
    id0  = (int *) malloc(num0*sizeof(int));
    p0    = (float *) malloc(num0*sizeof(float));
    dpdt0 = (float *) malloc(num0*sizeof(float));
    t0    = (float *) malloc(num0*sizeof(float));
    dtdt0 = (float *) malloc(num0*sizeof(float));
    s0    = (float *) malloc(num0*sizeof(float));
    dsdt0 = (float *) malloc(num0*sizeof(float));
    ps0   = (float *) malloc(num0*sizeof(float));
    pd0   = (float *) malloc(num0*sizeof(float));
    pt0   = (float *) malloc(num0*sizeof(float));
    ts0   = (float *) malloc(num0*sizeof(float));
    td0   = (float *) malloc(num0*sizeof(float));
    tt0   = (float *) malloc(num0*sizeof(float));
    ss0   = (float *) malloc(num0*sizeof(float));
    sd0   = (float *) malloc(num0*sizeof(float));
    st0   = (float *) malloc(num0*sizeof(float));
    je0   = (float *) malloc(num0*sizeof(float));
    lsda_read(handle,LSDA_INT,"ids",0,num0,id0);
  } else {
    num0 = 0;
  }
  lsda_queryvar(handle,"/jntforc/type1",&typid,&length,&filenum);
  if(typid == 0) {
    lsda_cd(handle,"/jntforc/type1/metadata");
    strcpy(title_dir,"/jntforc/type1/metadata");
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    num1 = length;
    id1   = (int *) malloc(num1*sizeof(int));
    a1    = (float *) malloc(num1*sizeof(float));
    dadt1 = (float *) malloc(num1*sizeof(float));
    g1    = (float *) malloc(num1*sizeof(float));
    dgdt1 = (float *) malloc(num1*sizeof(float));
    b1    = (float *) malloc(num1*sizeof(float));
    dbdt1 = (float *) malloc(num1*sizeof(float));
    as1   = (float *) malloc(num1*sizeof(float));
    ad1   = (float *) malloc(num1*sizeof(float));
    at1   = (float *) malloc(num1*sizeof(float));
    gsf1  = (float *) malloc(num1*sizeof(float));
    bs1   = (float *) malloc(num1*sizeof(float));
    bd1   = (float *) malloc(num1*sizeof(float));
    bt1   = (float *) malloc(num1*sizeof(float));
    je1   = (float *) malloc(num1*sizeof(float));
    jx1   = (float *) malloc(num1*sizeof(float));
    jx2   = (float *) malloc(num1*sizeof(float));
    lsda_read(handle,LSDA_INT,"ids",0,num1,id1);
  } else {
    num1 = 0;
  }
  if(strlen(title_dir)==0) return 0;  /* ?? */
/*
  open file and write header
*/
  sprintf(output_file,"%sjntforc",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_dir,fp);
  k=0;
  if(numj > 0) {
    lsda_cd(handle,"/jntforc/joints/metadata");
    i=(num0+num1 > 0) ? 0 : 1;
    output_legend(handle,fp,1,i);
    k = 1;
  }
  if(num0 > 0) {
    lsda_cd(handle,"/jntforc/type0/metadata");
    i=!k;
    j=(num1 > 0) ? 0 : 1;
    output_legend(handle,fp,i,j);
    k = 1;
  }
  if(num1 > 0) {
    lsda_cd(handle,"/jntforc/type1/metadata");
    i= !k;
    output_legend(handle,fp,i,1);
  }
/*
  Loop through time states and write each one.
*/
  for(state=1; ; state++) {
    if(numj > 0) {
      if(state<=999999)
        sprintf(dirname,"/jntforc/joints/d%6.6d",state);
      else
        sprintf(dirname,"/jntforc/joints/d%8.8d",state);

      lsda_queryvar(handle,dirname,&typid,&length,&filenum);
      if(typid != 0) break;
      lsda_cd(handle,dirname);
      if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"x_force",         0,numj,xf) != numj) break;
      if(lsda_read(handle,LSDA_FLOAT,"y_force",         0,numj,yf) != numj) break;
      if(lsda_read(handle,LSDA_FLOAT,"z_force",         0,numj,zf) != numj) break;
      if(lsda_read(handle,LSDA_FLOAT,"x_moment",        0,numj,xm) != numj) break;
      if(lsda_read(handle,LSDA_FLOAT,"y_moment",        0,numj,ym) != numj) break;
      if(lsda_read(handle,LSDA_FLOAT,"z_moment",        0,numj,zm) != numj) break;
      if(lsda_read(handle,LSDA_FLOAT,"resultant_force", 0,numj,rf) != numj) break;
      if(lsda_read(handle,LSDA_FLOAT,"resultant_moment",0,numj,rm) != numj) break;
      if(e && lsda_read(handle,LSDA_FLOAT,"energy",0,numj,e) != numj) break;
      fprintf(fp,"\n\n time.........................%14.5E\n",time);
      for(i=0; i<numj; i++) {
        fprintf(fp," joint ID.....................%10d\n",idj[i]);
        if(local[i]) {
          fprintf(fp," x-force  (local) ............%14.5E\n",xf[i]);
          fprintf(fp," y-force  (local) ............%14.5E\n",yf[i]);
          fprintf(fp," z-force  (local) ............%14.5E\n",zf[i]);
          fprintf(fp," x-moment (local) ............%14.5E\n",xm[i]);
          fprintf(fp," y-moment (local) ............%14.5E\n",ym[i]);
          fprintf(fp," z-moment (local) ............%14.5E\n",zm[i]);
        } else {
          fprintf(fp," x-force......................%14.5E\n",xf[i]);
          fprintf(fp," y-force......................%14.5E\n",yf[i]);
          fprintf(fp," z-force......................%14.5E\n",zf[i]);
          fprintf(fp," x-moment.....................%14.5E\n",xm[i]);
          fprintf(fp," y-moment.....................%14.5E\n",ym[i]);
          fprintf(fp," z-moment.....................%14.5E\n",zm[i]);
        }
        fprintf(fp," resultant force..............%14.5E\n",rf[i]);
        fprintf(fp," resultant moment.............%14.5E\n",rm[i]);
        if(e) fprintf(fp," energy.......................%14.5E\n",e[i]);
      }
    }
    if(num0 > 0) {
      if(state<=999999)
        sprintf(dirname,"/jntforc/type0/d%6.6d",state);
      else
        sprintf(dirname,"/jntforc/type0/d%8.8d",state);

      lsda_queryvar(handle,dirname,&typid,&length,&filenum);
      if(typid != 0) break;
      lsda_cd(handle,dirname);
      if(lsda_read(handle,LSDA_FLOAT,"phi_degrees",           0,num0,   p0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"d(phi)_dt",             0,num0,dpdt0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"theta_degrees",         0,num0,   t0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"d(theta)_dt",           0,num0,dtdt0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"psi_degrees",           0,num0,   s0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"d(psi)_dt",             0,num0,dsdt0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"phi_moment_stiffness",  0,num0,  ps0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"phi_moment_damping",    0,num0,  pd0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"phi_moment_total",      0,num0,  pt0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"theta_moment_stiffness",0,num0,  ts0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"theta_moment_damping",  0,num0,  td0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"theta_moment_total",    0,num0,  tt0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"psi_moment_stiffness",  0,num0,  ss0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"psi_moment_damping",    0,num0,  sd0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"psi_moment_total",      0,num0,  st0) != num0) break;
      if(lsda_read(handle,LSDA_FLOAT,"joint_energy",          0,num0,  je0) != num0) break;
      for(i=0; i<num0; i++) {
        fprintf(fp," joint stiffness id number....%10d\n",id0[i]);
        fprintf(fp," x-displacement...............%14.5E\n",p0[i]);
        fprintf(fp," d(dispx)/dt..................%14.5E\n",dpdt0[i]);
        fprintf(fp," y-displacement...............%14.5E\n",t0[i]);
        fprintf(fp," d(dispy)/dt..................%14.5E\n",dtdt0[i]);
        fprintf(fp," z-displacement...............%14.5E\n",s0[i]);
        fprintf(fp," d(dispz)/dt..................%14.5E\n",dsdt0[i]);
        fprintf(fp," force-x-stiffness............%14.5E\n",ps0[i]);
        fprintf(fp," force-x-damping..............%14.5E\n",pd0[i]);
        fprintf(fp," force-x-total................%14.5E\n",pt0[i]);
        fprintf(fp," force-y-stiffness............%14.5E\n",ts0[i]);
        fprintf(fp," force-y-damping..............%14.5E\n",td0[i]);
        fprintf(fp," force-y-total................%14.5E\n",tt0[i]);
        fprintf(fp," force-z-stiffness............%14.5E\n",ss0[i]);
        fprintf(fp," force-z-damping..............%14.5E\n",sd0[i]);
        fprintf(fp," force-z-total................%14.5E\n",st0[i]);
        fprintf(fp," joint energy.................%14.5E\n",je0[i]);
      }
    }
    if(num1 > 0) {
      if(state<=999999)
        sprintf(dirname,"/jntforc/type1/d%6.6d",state);
      else
        sprintf(dirname,"/jntforc/type1/d%8.8d",state);

      lsda_queryvar(handle,dirname,&typid,&length,&filenum);
      if(typid != 0) break;
      lsda_cd(handle,dirname);
      if(numj == 0 & num0 ==0) {
        if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
        fprintf(fp,"\n\n time.........................%14.5E\n",time);
      }
      if(lsda_read(handle,LSDA_FLOAT,"alpha_degrees",         0,num1,   a1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"d(alpha)_dt",           0,num1,dadt1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"gamma_degrees",         0,num1,   g1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"d(gamma)_dt",           0,num1,dgdt1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"beta_degrees",          0,num1,   b1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"d(beta)_dt",            0,num1,dbdt1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"alpha_moment_stiffness",0,num1,  as1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"alpha_moment_damping",  0,num1,  ad1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"alpha_moment_total",    0,num1,  at1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"gamma_scale_factor",    0,num1, gsf1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"beta_moment_stiffness", 0,num1,  bs1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"beta_moment_damping",   0,num1,  bd1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"beta_moment_total",     0,num1,  bt1) != num1) break;
      if(lsda_read(handle,LSDA_FLOAT,"joint_energy",          0,num1,  je1) != num1) break;
/*
  This is such a hack I don't know where to begin.....
  Someone somewhere decided that instead of creating a new
  output type (like they should have) they would override
  the meaning of all these arrays depending on the output
  of these two "extra" variables.  I will change this when
  I get a chance, but for now we have to support this for
  backward compatibility.  Also, these two may not be there
  at all, so we have to allow for that too.
*/
      if(have_type1_extra == -1) {
        lsda_queryvar(handle,"joint_extra1",&typid,&length,&filenum);
        if(typid > 0)
          have_type1_extra = 1;
        else
          have_type1_extra = 0;
      }
      if(have_type1_extra) {
      if(lsda_read(handle,LSDA_FLOAT,"joint_extra1",          0,num1,  jx1) != num1) break;
        if(lsda_read(handle,LSDA_FLOAT,"joint_extra2",          0,num1,  jx2) != num1) break;
      }
      for(i=0; i<num1; i++) {
        if (have_type1_extra && jx1[i]+jx2[i] != -8){
          fprintf(fp," joint stiffness id number....%10d\n",id1[i]);
          fprintf(fp," phi (degrees)................%14.5E\n",a1[i]);
          fprintf(fp," d(phi)/dt (degrees)..........%14.5E\n",dadt1[i]);
          fprintf(fp," theta (degrees)..............%14.5E\n",g1[i]);
          fprintf(fp," d(theta)/dt (degrees)........%14.5E\n",dgdt1[i]);
          fprintf(fp," psi (degrees)................%14.5E\n",b1[i]);
          fprintf(fp," d(psi)/dt (degrees)..........%14.5E\n",dbdt1[i]);
          fprintf(fp," phi moment-stiffness.........%14.5E\n",as1[i]);
          fprintf(fp," phi moment-damping...........%14.5E\n",ad1[i]);
          fprintf(fp," phi moment-total.............%14.5E\n",at1[i]);
          fprintf(fp," theta-moment-stiffness.......%14.5E\n",gsf1[i]);
          fprintf(fp," theta-moment-damping.........%14.5E\n",bs1[i]);
          fprintf(fp," theta-moment-total...........%14.5E\n",bd1[i]);
          fprintf(fp," psi-moment-stiffness.........%14.5E\n",bt1[i]);
          fprintf(fp," psi-moment-damping...........%14.5E\n",je1[i]);
          fprintf(fp," psi-moment-total.............%14.5E\n",jx1[i]);
          fprintf(fp," joint energy.................%14.5E\n",jx2[i]);
        } else {
          fprintf(fp," joint stiffness id number....%10d\n",id1[i]);
          fprintf(fp," alpha (degrees)..............%14.5E\n",a1[i]);
          fprintf(fp," d(alpha)/dt (degrees)........%14.5E\n",dadt1[i]);
          fprintf(fp," gamma (degrees)..............%14.5E\n",g1[i]);
          fprintf(fp," d(gamma)/dt (degrees)........%14.5E\n",dgdt1[i]);
          fprintf(fp," beta (degrees)...............%14.5E\n",b1[i]);
          fprintf(fp," d(beta)/dt (degrees).........%14.5E\n",dbdt1[i]);
          fprintf(fp," alpha-moment-stiffness.......%14.5E\n",as1[i]);
          fprintf(fp," alpha-moment-damping.........%14.5E\n",ad1[i]);
          fprintf(fp," alpha-moment-total...........%14.5E\n",at1[i]);
          fprintf(fp," gamma scale factor...........%14.5E\n",gsf1[i]);
          fprintf(fp," beta-moment-stiffness........%14.5E\n",bs1[i]);
          fprintf(fp," beta-moment-damping..........%14.5E\n",bd1[i]);
          fprintf(fp," beta-moment-total............%14.5E\n",bt1[i]);
          fprintf(fp," joint energy.................%14.5E\n",je1[i]);
        }
      }
    }
  }
  fclose(fp);
  if(num1 > 0) {
    free(je1);
    free(bt1);
    free(bd1);
    free(bs1);
    free(gsf1);
    free(at1);
    free(ad1);
    free(as1);
    free(dbdt1);
    free(b1);
    free(dgdt1);
    free(g1);
    free(dadt1);
    free(a1);
    free(id1);
  }
  if(num0 > 0) {
    free(je0);
    free(st0);
    free(sd0);
    free(ss0);
    free(tt0);
    free(td0);
    free(ts0);
    free(pt0);
    free(pd0);
    free(ps0);
    free(dsdt0);
    free(s0);
    free(dtdt0);
    free(t0);
    free(dpdt0);
    free(p0);
    free(id0);
  }
  if(numj > 0) {
    if(e) free(e);
    free(rm);
    free(rf);
    free(zm);
    free(ym);
    free(xm);
    free(zf);
    free(yf);
    free(xf);
    free(local);
    free(idj);
  }
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  SPHOUT file
*/
int translate_sphout(int handle)
{
  int i,j,k,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[128];
  int *ids,*mat;
  int cycle;
  float time;
  float *sig_xx,*sig_yy,*sig_zz;
  float *sig_xy,*sig_yz,*sig_zx;
  float *eps_xx,*eps_yy,*eps_zz;
  float *eps_xy,*eps_yz,*eps_zx;
  float *density,*smooth,*temp;
  int *neigh,*act,*nstate;
  float *yield,*effsg;
  char states[5][16];
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/sphout/metadata") == -1) return 0;
  printf("Extracting SPHOUT data\n");
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num=length;


/*
  allocate memory to read in 1 state
*/
  ids    = (int *) malloc(num*sizeof(int));
  mat    = (int *) malloc(num*sizeof(int));
  sig_xx = (float *) malloc(num*sizeof(float));
  sig_yy = (float *) malloc(num*sizeof(float));
  sig_zz = (float *) malloc(num*sizeof(float));
  sig_xy = (float *) malloc(num*sizeof(float));
  sig_yz = (float *) malloc(num*sizeof(float));
  sig_zx = (float *) malloc(num*sizeof(float));
  eps_xx = (float *) malloc(num*sizeof(float));
  eps_yy = (float *) malloc(num*sizeof(float));
  eps_zz = (float *) malloc(num*sizeof(float));
  eps_xy = (float *) malloc(num*sizeof(float));
  eps_yz = (float *) malloc(num*sizeof(float));
  eps_zx = (float *) malloc(num*sizeof(float));
  density= (float *) malloc(num*sizeof(float));
  smooth = (float *) malloc(num*sizeof(float));
  yield  = (float *) malloc(num*sizeof(float));
  effsg  = (float *) malloc(num*sizeof(float));
  act    = (int *) malloc(num*sizeof(int));
  temp   = (float *) malloc(num*sizeof(float));
  neigh  = (int *) malloc(num*sizeof(int));
  nstate = (int *) malloc(num*sizeof(int));

/*
  Read metadata
*/
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
  lsda_read(handle,LSDA_INT,"mat",0,num,mat);
  lsda_queryvar(handle,"states",&typid,&length,&filenum);
  lsda_read(handle,LSDA_I1,"states",0,length,dirname);
  for(i=j=k=0; i<length; i++) {
    if(dirname[i] == ',') {
      states[j][k]=0;
      j++;
      k=0;
    } else {
      states[j][k++]=dirname[i];
    }
  }
  states[j][k]=0;
/*
  open file and write header
*/
  sprintf(output_file,"%ssphout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/sphout/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/sphout",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xx",0,num,sig_xx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yy",0,num,sig_yy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zz",0,num,sig_zz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sig_xy",0,num,sig_xy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sig_yz",0,num,sig_yz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sig_zx",0,num,sig_zx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"eps_xx",0,num,eps_xx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yy",0,num,eps_yy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zz",0,num,eps_zz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"eps_xy",0,num,eps_xy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"eps_yz",0,num,eps_yz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"eps_zx",0,num,eps_zx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"density",0,num,density) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"smooth",0,num,smooth) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"yield",0,num,yield) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"effsg",0,num,effsg) != num) break;
    if(lsda_read(handle,LSDA_INT,"neigh",0,num,neigh) != num) break;
    if(lsda_read(handle,LSDA_INT,"act",0,num,act) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"temperature",0,num,temp) != num) break;
    if(lsda_read(handle,LSDA_INT,"state",0,num,nstate) != num) break;

    fprintf(fp,"\n\n\n S P H   o u t p u t       at time%11.4E",time);
    fprintf(fp,"     for  time  step%10d\n",cycle);
    fprintf(fp,"\n particle  mat    sig-xx      sig-yy      sig-zz");
    fprintf(fp,"      sig-xy      sig-yz      sig-zx  \n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%8d%6d",ids[i],mat[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
          sig_xx[i],sig_yy[i],sig_zz[i],sig_xy[i],sig_yz[i],sig_zx[i]);
    }
    fprintf(fp,"\n particle  mat    eps-xx      eps-yy      eps-zz");
    fprintf(fp,"      eps-xy      eps-yz      eps-zx \n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%8d%6d",ids[i],mat[i]);
      fprintf(fp,"%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",
          eps_xx[i],eps_yy[i],eps_zz[i],eps_xy[i],eps_yz[i],eps_zx[i]);
    }
    fprintf(fp,"\n particle  mat    density     smooth     neigh   act  temperature\n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%8d%6d",ids[i],mat[i]);
      fprintf(fp,"%12.4E%12.4E%7d%6d  %12.4E\n",density[i],smooth[i],neigh[i],act[i],temp[i]);
    }
    fprintf(fp,"\n particle  mat    yield       effsg       state \n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%8d%6d",ids[i],mat[i]);
      fprintf(fp,"%12.4E%12.4E   %7s\n",yield[i],effsg[i],states[nstate[i]-1]);
    }
  }
  fclose(fp);
  free(ids);
  free(mat);
  free(sig_xx);
  free(sig_yy);
  free(sig_zz);
  free(sig_xy);
  free(sig_yz);
  free(sig_zx);
  free(eps_xx);
  free(eps_yy);
  free(eps_zz);
  free(eps_xy);
  free(eps_yz);
  free(eps_zx);
  free(density);
  free(smooth);
  free(yield);
  free(effsg);
  free(act);
  free(temp);
  free(neigh);
  free(nstate);
  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  DEFGEO file
*/
int translate_defgeo(int handle)
{
  int j,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int *ids;
  float *dx,*dy,*dz;
  float maxdisp;
  int cycle;
  FILE *fp;
  char *defgeoenv, *outtype;
  LSDADir *dp = NULL;
/*
   now try to find out which format should output
   LSTC_DEFGEO 0        - ls-dyna format
               chrysler - Chrysler format
*/
defgeoenv = (char *) malloc(20);
outtype   = (char *) malloc(20);

defgeoenv = "LSTC_DEFGEO";
outtype   = (char *) getenv(defgeoenv);

  if (lsda_cd(handle,"/defgeo/metadata") == -1) return 0;
  printf("Extracting DEFGEO data - ");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  ids          = (int   *) malloc(num*sizeof(int));
  dx           = (float *) malloc(num*sizeof(float));
  dy           = (float *) malloc(num*sizeof(float));
  dz           = (float *) malloc(num*sizeof(float));
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%sdefgeo",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
/*
  output_title(handle,"/defgeo/metadata",fp);
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/defgeo",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"x_displacement",0,num,dx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_displacement",0,num,dy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_displacement",0,num,dz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"max_displacement",0,1,&maxdisp) != 1) break;

    cycle = 1000 + state - 1;
    if(outtype!=NULL && (outtype[0]=='c' || outtype[0]=='C' || outtype[0]=='1')) {
    /* Chrysler format */
    if(state==1) {
    printf("Chrysler format\n");
    fprintf(fp,"$LABEL\n");
    fprintf(fp,"LS      DYNA3D              89P1\n");
    }

    fprintf(fp,"$DISPLACEMENTS\n\n\n\n");
    fprintf(fp,"%8d%8d%8d       0%8d       1       3       1       1\n",
    (int)cycle,(int)num,(int)cycle,(int)cycle);
    fprintf(fp,"%16.7E%16.7E%16.7E%16.7E\n",maxdisp,0.,0.,0.);
    for(j=0; j<num; j++)
      fprintf(fp,"%8d%16.7E%16.7E%16.7E\n",(int)ids[j],dx[j],dy[j],dz[j]);
  } else {
  /* LS-DYNA format */
  if(state==1) printf("LS-DYNA format\n");
    fprintf(fp,"  6000     1%6d                                              %8d\n",
      (int)cycle,(int)num);

    for(j=0; j<num; j++)
      fprintf(fp,"%8d%8.2g%8.2g%8.2g\n",(int)ids[j],dx[j],dy[j],dz[j]);
  }
  }
  fclose(fp);
  free(dx);
  free(dy);
  free(dz);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 1;
}
/*
  DCFAIL file
*/
int translate_dcfail(int handle)
{
  int i,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[256];
  int *ids, *type;
  int have_torsion;
  float *area,*bend,*rate,*fail,*normal,*shear,*area_sol;
  float *axial_f,*shear_f,*torsion_m,*bending_m;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/dcfail/metadata") == -1) return 0;
  printf("Extracting DCFAIL data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  ids = (int *) malloc(num*sizeof(int));
  type = (int *) malloc(num*sizeof(int));
  area  = (float *) malloc(11*num*sizeof(float));
  bend  = area   + num;
  rate  = bend   + num;
  fail = rate   + num;
  normal = fail  + num;
  shear = normal  + num;
  area_sol = shear  + num;
  axial_f = area_sol + num;
  shear_f = axial_f + num;
  torsion_m = shear_f + num;
  bending_m = torsion_m + num;

  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
  lsda_read(handle,LSDA_INT,"type",0,num,type);
/*
  open file and write header
*/
  sprintf(output_file,"%sdcfail",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/dcfail/metadata",fp);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/dcfail",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"area",0,num,area) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"bending_term",0,num,bend) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"effective_strain_rate",0,num,rate) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"failure_function",0,num,fail) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"normal_term",0,num,normal) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"shear_term",0,num,shear) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"area_sol",0,num,area_sol) != num) break;
    have_torsion=1;
    if(lsda_read(handle,LSDA_FLOAT,"axial_force",0,num,axial_f) != num) have_torsion=0;
    if(lsda_read(handle,LSDA_FLOAT,"shear_force",0,num,shear_f) != num) have_torsion=0;
    if(lsda_read(handle,LSDA_FLOAT,"torsional_moment",0,num,torsion_m) != num) have_torsion=0;
    if(lsda_read(handle,LSDA_FLOAT,"bending_moment",0,num,bending_m) != num) have_torsion=0;

    fprintf(fp,"\n define connection failure, time=%13.5E",time);
    fprintf(fp,"\n    ele_id    con_id    fail_func       normal  ");
    fprintf(fp,"    bending        shear         area     area_sol   eff_e_rate");
    if(have_torsion)
      fprintf(fp,"  axial force  shear force  torsion mom  bending mom");
    fprintf(fp,"\n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%10d%10d",ids[i],type[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E",fail[i],normal[i],bend[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E",shear[i],area[i],area_sol[i],rate[i]);
      if(have_torsion)
        fprintf(fp,"%13.5E%13.5E%13.5E%13.5E",axial_f[i],shear_f[i],torsion_m[i],bending_m[i]);
      fprintf(fp,"\n");
    }

  }
  fclose(fp);
  free(area);
  free(type);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  TPRINT file
*/
int translate_tprint(int handle)
{
  int i,typid,num,num0,filenum,state;
  LSDA_Length length;
  char dirname[128];
  int *ids,maxnode,minnode,nstep,iteration,numsh12,numnp;
  float maxtemp,mintemp,*temp,temp_norm;
  float time,timestep,*t_bottom,*t_top;
  float *x_flux,*y_flux,*z_flux;
  FILE *fp;
  int *mx,nesum,itran,nummat,nbs,nfbc,ncbc,nrbc,nebc,nbnseg;
  int nthssf,nthssc,nthssr,nthsse,*idssfi,*idssci,*idssri,*idssei,nsl2d,nsl3d,*conid3d;
  float *dqgen,*qgen,*dh,*th,*esum,*sumf,*sumfdt,*sumc,*sume,*sumedt;
  float *sumcdt,*sumr,*sumrdt,*ebal2d,*ebal2ddt,*ebal3d,*ebal3ddt;
  LSDADir *dp = NULL;

  sprintf(dirname,"/tprint/d000001");
  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
  if(typid != 0) return 0;
  lsda_cd(handle,dirname);
  printf("Extracting TPRINT data\n");

  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num=length;
  num0=num;
/*
  allocate memory to read in 1 state
*/
  ids    = (int *) malloc(num*sizeof(int));
  temp   = (float *) malloc(num*sizeof(float));
  x_flux = (float *) malloc(num*sizeof(float));
  y_flux = (float *) malloc(num*sizeof(float));
  z_flux = (float *) malloc(num*sizeof(float));
  lsda_queryvar(handle,"t_top",&typid,&length,&filenum);
  if(typid > 0) {
    t_top = (float *) malloc(num*sizeof(float));
    t_bottom = (float *) malloc(num*sizeof(float));
  } else {
     t_top = t_bottom = NULL;
  }
  lsda_queryvar(handle,"energy sum",&typid,&length,&filenum);
  nesum=length;
  esum = (float *) malloc(nesum*sizeof(int)); 
/*
  open file and write header
*/
  sprintf(output_file,"%stprint",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/tprint/metadata",fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/tprint",dp,dirname)) != NULL; state++) {
    lsda_queryvar(handle,"ids",&typid,&length,&filenum);
    num=length;
    if(num > num0) {
      free(z_flux);
      free(y_flux);
      free(x_flux);
      free(temp);
      free(ids);
      ids    = (int *) malloc(num*sizeof(int));
      temp   = (float *) malloc(num*sizeof(float));
      x_flux = (float *) malloc(num*sizeof(float));
      y_flux = (float *) malloc(num*sizeof(float));
      z_flux = (float *) malloc(num*sizeof(float));
      if(t_top) {
        free(t_top);
        free(t_bottom);
        t_top    = (float *) malloc(num*sizeof(float));
        t_bottom = (float *) malloc(num*sizeof(float));
      }
      num0 = num;
    }
    if(lsda_read(handle,LSDA_INT,"ids",0,num,ids) != num) break;
    if(lsda_read(handle,LSDA_INT,"maxnode",0,1,&maxnode) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"maxtemp",0,1,&maxtemp) != 1) break;
    if(lsda_read(handle,LSDA_INT,"minnode",0,1,&minnode) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"mintemp",0,1,&mintemp) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nstep",0,1,&nstep) != 1) break;
    if(lsda_read(handle,LSDA_INT,"solution_iterations",0,1,&iteration) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"temperature",0,num,temp) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"temperature_norm",0,1,&temp_norm) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"timestep",0,1,&timestep) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_flux",0,num,x_flux) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_flux",0,num,y_flux) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_flux",0,num,z_flux) != num) break;
    if(lsda_read(handle,LSDA_INT,"itran",0,1,&itran) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nbs",0,1,&nbs) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nfbc",0,1,&nfbc) != 1) break;
    if(lsda_read(handle,LSDA_INT,"ncbc",0,1,&ncbc) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nrbc",0,1,&nrbc) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nebc",0,1,&nebc) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nbnseg",0,1,&nbnseg) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nthssf",0,1,&nthssf) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nthssc",0,1,&nthssc) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nthssr",0,1,&nthssr) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nthsse",0,1,&nthsse) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nsl2d",0,1,&nsl2d) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nsl3d",0,1,&nsl3d) != 1) break;
    if(lsda_read(handle,LSDA_INT,"numnp",0,1,&numnp) != 1) break;
    if(lsda_read(handle,LSDA_INT,"numsh12",0,1,&numsh12) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"energy sum",0,nesum,esum) != nesum) break;
    if (nsl2d>0) {
      ebal2d   = (float *) malloc(nsl2d*sizeof(int));
      ebal2ddt = (float *) malloc(nsl2d*sizeof(int));
      if(lsda_read(handle,LSDA_FLOAT,"ebal2d",0,nsl2d,ebal2d) != nsl2d) break;
      if(lsda_read(handle,LSDA_FLOAT,"ebal2ddt",0,nsl2d,ebal2ddt) != nsl2d) break;
    }
    if (nsl3d>0) {
      conid3d  = (int   *) malloc(nsl3d*sizeof(int));
      ebal3d   = (float *) malloc(nsl3d*sizeof(int));
      ebal3ddt = (float *) malloc(nsl3d*sizeof(int));
      if(lsda_read(handle,LSDA_INT,"conid3d",0,nsl3d,conid3d) != nsl3d) break;
      if(lsda_read(handle,LSDA_FLOAT,"ebal3d",0,nsl3d,ebal3d) != nsl3d) break;
      if(lsda_read(handle,LSDA_FLOAT,"ebal3ddt",0,nsl3d,ebal3ddt) != nsl3d) break;
    }
    if (nbs>0) {
      if (nfbc>0) {
        if (nthssf!=0) {
        idssfi = (int   *) malloc(nthssf*sizeof(int));
        sumf   = (float *) malloc(nthssf*sizeof(int));
        sumfdt = (float *) malloc(nthssf*sizeof(int));
        if(lsda_read(handle,LSDA_INT,"idssfi",0,nthssf,idssfi) != nthssf) break;
        if(lsda_read(handle,LSDA_FLOAT,"sumf",0,nthssf,sumf) != nthssf) break;
        if(lsda_read(handle,LSDA_FLOAT,"sumfdt",0,nthssf,sumfdt) != nthssf) break; }
      }
      if (ncbc>0) {
        if (nthssc!=0) {
        idssci = (int   *) malloc(nthssc*sizeof(int));
        sumc   = (float *) malloc(nthssc*sizeof(int));
        sumcdt = (float *) malloc(nthssc*sizeof(int)); 
        if(lsda_read(handle,LSDA_INT,"idssci",0,nthssc,idssci) != nthssc) break;
        if(lsda_read(handle,LSDA_FLOAT,"sumc",0,nthssc,sumc) != nthssc) break;
        if(lsda_read(handle,LSDA_FLOAT,"sumcdt",0,nthssc,sumcdt) != nthssc) break; }
      }
      if (nrbc>0) {
        if (nthssr!=0) {
        idssri = (int   *) malloc(nthssc*sizeof(int));
        sumr   = (float *) malloc(nthssr*sizeof(int));
        sumrdt = (float *) malloc(nthssr*sizeof(int)); 
        if(lsda_read(handle,LSDA_INT,"idssri",0,nthssr,idssri) != nthssr) break;
        if(lsda_read(handle,LSDA_FLOAT,"sumr",0,nthssr,sumr) != nthssr) break;
        if(lsda_read(handle,LSDA_FLOAT,"sumrdt",0,nthssr,sumrdt) != nthssr) break; } 
      }
      if (nebc>0) {
        if (nthsse!=0) {
        idssei = (int   *) malloc(nthsse*sizeof(int));
        sume   = (float *) malloc(nthsse*sizeof(int));
        sumedt = (float *) malloc(nthsse*sizeof(int)); 
        if(lsda_read(handle,LSDA_INT,"idssei",0,nthsse,idssei) != nthsse) break;
        if(lsda_read(handle,LSDA_FLOAT,"sume",0,nthsse,sume) != nthsse) break;
        if(lsda_read(handle,LSDA_FLOAT,"sumedt",0,nthsse,sumedt) != nthsse) break; }
      }
    }
    if(t_top) {
      if(lsda_read(handle,LSDA_FLOAT,"t_top",0,num,t_top) != num) break;
      if(lsda_read(handle,LSDA_FLOAT,"t_bottom",0,num,t_bottom) != num) break;
    }
    if(lsda_read(handle,LSDA_INT,"nummat",0,1,&nummat) != 1) break;
    mx = (int *) malloc(nummat*sizeof(int));
    dqgen = (float *) malloc(nummat*sizeof(int));
    qgen = (float *) malloc(nummat*sizeof(int));
    dh = (float *) malloc(nummat*sizeof(int));
    th = (float *) malloc(nummat*sizeof(int));
    if(lsda_read(handle,LSDA_INT,"mat ids",0,nummat,mx) != nummat) break; 
    if(lsda_read(handle,LSDA_FLOAT,"heat generated",0,nummat,dqgen) != nummat) break; 
    if(lsda_read(handle,LSDA_FLOAT,"total heat gen",0,nummat,qgen) != nummat) break; 
    if(lsda_read(handle,LSDA_FLOAT,"energy change",0,nummat,dh) != nummat) break; 
    if(lsda_read(handle,LSDA_FLOAT,"total energy change",0,nummat,th) != nummat) break; 

    fprintf(fp," ****************************************");
    fprintf(fp,"******************************\n\n");
    if (itran!=0) {
    fprintf(fp,"     time =%12.4E     time step =%12.4E     thermal step no.=%6d\n\n",
      time,timestep,nstep);
      } else {
        fprintf(fp," steady state solution\n\n");
    }
    fprintf(fp,"     minimum temperature = %12.4E   at node %8d\n",mintemp,minnode);
    fprintf(fp,"     maximum temperature = %12.4E   at node %8d\n\n",maxtemp,maxnode);
    if(t_top) {
      fprintf(fp,"    node temperature    t-bottom       t-top      x-flux      y-flux      z-flux\n\n");
      for(i=0; i<num; i++) {
/* check for valid data -- only some nodes have actual data here */
        if(t_bottom[i] > -1.e+10) {
          fprintf(fp,"%8d%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",ids[i],temp[i],t_bottom[i],t_top[i],x_flux[i],y_flux[i],z_flux[i]);
        } else {
          fprintf(fp,"%8d%12.4E                        %12.4E%12.4E%12.4E\n",ids[i],temp[i],x_flux[i],y_flux[i],z_flux[i]);
        }
      }
    } else {
      fprintf(fp,"*TEMPERATURE_NODE     numnode=%10d   thkshl=%10d\n",numnp,numsh12);
      fprintf(fp,"  node   temperature      x-flux      y-flux      z-flux\n\n");
      for(i=0; i<num; i++) {
        fprintf(fp,"%8d%12.4E%12.4E%12.4E%12.4E\n",ids[i],temp[i],x_flux[i],y_flux[i],z_flux[i]);
      }
    }
    if (nbs!=0) {
      fprintf(fp,"\n Boundary condition segment set heat transfer rate [energy/time]\n");
      fprintf(fp," Positive heat flow is in direction of the surface outward normal vector\n");
      fprintf(fp," NOTE: energy sum may not be 0. This depends on the model definition\n");
      fprintf(fp," Such as, surfaces with temperature boundary conditions are not included.\n");
      fprintf(fp,"\n*ENERGY_BOUNDARY         nfbc=%10d     ncbc=%10d     nrbc=%10d\n",nthssf,nthssc,nthssr);
      fprintf(fp,"     bc type   seg set   [energy/time]        [energy]\n");
      if (nfbc>0) {
          for(i=0; i<nthssf; i++) {
          fprintf(fp,"  flux      %10d%16.4E%16.4E\n",idssfi[i],sumf[i],sumfdt[i]);
        }
      }
      if (ncbc>0) {
        for(i=0; i<nthssc; i++) {
          fprintf(fp,"  convection%10d%16.4E%16.4E\n",idssci[i],sumc[i],sumcdt[i]);
        }
      }
      if (nrbc>0) {
        for(i=0; i<nthssr; i++) {
          fprintf(fp,"  radiation %10d%16.4E%16.4E\n",idssri[i],sumr[i],sumrdt[i]);
        }
      }
      if (nebc>0) {
        for(i=0; i<nthsse; i++) {
          fprintf(fp,"    enclosure        %8d       %12.4E  %12.4E\n",idssei[i],sume[i],sumedt[i]);
        }
      }
    }

    fprintf(fp,"                           -----------     -----------\n");
    fprintf(fp,"              energy sum =%12.4E    %12.4E\n\n",esum[0],esum[1]);

    if (nsl2d!=0 || nsl3d!=0) {
      fprintf(fp,"\n contact surface heat transfer rate\n");
      fprintf(fp," positive heat flow is in direction of master-->slave\n");
      if (nsl2d>0) fprintf(fp,"\n*ENERGY_CONTACT         ncont=%10d\n",nsl2d);
      if (nsl3d>0) fprintf(fp,"\n*ENERGY_CONTACT         ncont=%10d\n",nsl3d);
      if (nsl2d>0) fprintf(fp,"    order#   [energy/time]        [energy]\n");
      if (nsl3d>0) fprintf(fp,"    order# contact_id  [energy/time]        [energy]\n");
      if (nsl2d!=0) {
        for(i=0; i<nsl2d; i++) {
          fprintf(fp,"%10d%16.4E%16.4E\n",i+1,ebal2d[i],ebal2ddt[i]); 
        }
      }
      if (nsl3d!=0) {
        for(i=0; i<nsl3d; i++) {
          fprintf(fp,"%10d%10d%16.4E%16.4E\n",i+1,conid3d[i],ebal3d[i],ebal3ddt[i]); 
        }
      }
    }

    fprintf(fp,"\n*ENERGY_PART            npart=%10d\n",nummat);
    fprintf(fp,"  part #  Qgen_chg[energy]  Qgen_tot[energy] IE_chg[energy]  IE_tot[energy]\n");
      for(i=0; i<nummat; i++) {
        fprintf(fp,"%8d%16.4E%16.4E%16.4E%16.4E\n",mx[i],dqgen[i],qgen[i],dh[i],th[i]); 
      } 
    fprintf(fp,"\n number of solution iterations  =%5d\n\n",iteration);
    fprintf(fp,  " QA temperature norm, Tmin, Tmax= %12.4E %12.4E %12.4E\n\n",temp_norm,mintemp,maxtemp);
  }
  fclose(fp);
  free(z_flux);
  free(y_flux);
  free(x_flux);
  free(temp);
  free(ids);
  if (nfbc>0) {
    free(idssfi);
    free(sumf);
    free(sumfdt);
  }
  if (ncbc>0) {
    free(idssci);
    free(sumc);
    free(sumcdt);
  }
  if (nrbc>0) {
    free(idssri);
    free(sumr);
    free(sumrdt);
  }
  if (nebc>0) {
    free(idssei);
    free(sume);
    free(sumedt);
  }
  if (nummat>0) {
    free(mx);
    free(dqgen);
    free(qgen);
    free(dh);
    free(th);
  }
  if (nesum>0) {
    free(esum);
  }
  if (nsl2d>0) {
    free(ebal2d);
    free(ebal2ddt);
  }
  if (nsl3d>0) {
    free(conid3d);
    free(ebal3d);
    free(ebal3ddt);
  }
  if(t_top) {
    free(t_top);
    free(t_bottom);
  }
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  TRHIST file
*/
int translate_trhist(int handle)
{
  int i,typid,num,num0,filenum,state;
  LSDA_Length length;
  char dirname[128];
  int maxnode,minnode,nstep,iteration;
  float maxtemp,mintemp,*temp,temp_norm;
  float time,timestep,*t_bottom,*t_top;
  float *fiop,*x,*y,*z,*vx,*vy,*vz,*sx,*sy,*sz,*sxy,*syz,*szx,*efp,*rvl,*rho;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/trhist/metadata") == -1) return 0;
  printf("Extracting TRHIST data\n");

  sprintf(dirname,"/trhist/d000001");
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"fiop",&typid,&length,&filenum);
  num=length;
/*
  allocate memory to read in 1 state
*/
  fiop   = (float *) malloc(num*sizeof(float));
  x      = (float *) malloc(num*sizeof(float));
  y      = (float *) malloc(num*sizeof(float));
  z      = (float *) malloc(num*sizeof(float));
  vx     = (float *) malloc(num*sizeof(float));
  vy     = (float *) malloc(num*sizeof(float));
  vz     = (float *) malloc(num*sizeof(float));
  sx     = (float *) malloc(num*sizeof(float));
  sy     = (float *) malloc(num*sizeof(float));
  sz     = (float *) malloc(num*sizeof(float));
  sxy    = (float *) malloc(num*sizeof(float));
  syz    = (float *) malloc(num*sizeof(float));
  szx    = (float *) malloc(num*sizeof(float));
  efp    = (float *) malloc(num*sizeof(float));
  rvl    = (float *) malloc(num*sizeof(float));
  rho    = (float *) malloc(num*sizeof(float));
/*
  open file and write header
*/
  sprintf(output_file,"%strhist",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;

  fprintf(fp,"Tracer particle file\n");
  fprintf(fp,"%5d   16\n",num);
  fprintf(fp,"            x            y            z           vx           vy           vz\n");
  fprintf(fp,"           sx           sy           sz          sxy          syz          szx\n");
  fprintf(fp,"          efp          rho         rvol       active\n");
/*
  output_title(handle,"/trhist/metadata",fp);
  output_legend(handle,fp,1,1);
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/trhist",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"fiop",0,num,fiop) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x",0,num,x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y",0,num,y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z",0,num,z) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vx",0,num,vx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vy",0,num,vy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vz",0,num,vz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sx",0,num,sx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sy",0,num,sy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sz",0,num,sz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sxy",0,num,sxy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"syz",0,num,syz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"szx",0,num,szx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"efp",0,num,efp) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rvl",0,num,rvl) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rho",0,num,rho) != num) break;

    fprintf(fp,"%13.5E\n",time);
    for (i=0;i<num;i++) {
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",x[i],y[i],z[i],vx[i],vy[i],vz[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",sx[i],sy[i],sz[i],sxy[i],syz[i],szx[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E\n",efp[i],rho[i],rvl[i],fiop[i]);
    }
  }
  fclose(fp);

  free(rho);
  free(rvl);
  free(efp);
  free(szx);
  free(syz);
  free(sxy);
  free(sz);
  free(sy);
  free(sx);
  free(vz);
  free(vy);
  free(vx);
  free(z);
  free(y);
  free(x);
  free(fiop);

  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  ALE DBSENSOR file
*/
int translate_dbsensor(int handle)
{
  int i,typid,num,num0,filenum,state;
  LSDA_Length length;
  char dirname[128];
  float time,timestep;
  float *x,*y,*z,*pres,*temp;
  int   *ids,*id_solid;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/dbsensor/metadata") == -1) return 0;
  printf("Extracting DBSENSOR data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  ids          = (int   *) malloc(num*sizeof(int));
  x            = (float *) malloc(num*sizeof(float));
  y            = (float *) malloc(num*sizeof(float));
  z            = (float *) malloc(num*sizeof(float));
  pres         = (float *) malloc(num*sizeof(float));
  temp         = (float *) malloc(num*sizeof(float));
  id_solid     = (int   *) malloc(num*sizeof(int));
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%sdbsensor",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;

  fprintf(fp,"     ALE sensors output\n");
  fprintf(fp,"     Number of sensors:%10d\n\n\n\n",num);
  fprintf(fp,"        id          x            y             z             p    Cpld Solid");
  for(state=1; (dp = next_dir(handle,"/dbsensor",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x",0,num,x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y",0,num,y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z",0,num,z) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"pressure",0,num,pres) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"temperature",0,num,temp) != num) break;
    if(lsda_read(handle,LSDA_INT  ,"solid_id",0,num,id_solid) != num) break;

    fprintf(fp,"\n   time=%13.5E\n",time);
    for (i=0;i<num;i++) {
      fprintf(fp,"%10d%14.4E%14.4E%14.4E%14.4E%10d%14.4E\n",ids[i],x[i],y[i],z[i],pres[i],id_solid[i],temp[i]);
    }
  }
  fclose(fp);

  free(id_solid);
  free(temp);
  free(pres);
  free(z);
  free(y);
  free(x);
  free(ids);

  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  ALE FSI file
*/
int translate_dbfsi(int handle)
{
  int i,typid,num,num0,filenum,state;
  LSDA_Length length;
  char dirname[128];
  float time,timestep;
  float *fx,*fy,*fz,*pres,*pleak,*flux,*gx,*gy,*gz,*Ptmp,*PDt;
  int   *ids;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/dbfsi/metadata") == -1) return 0;
  printf("Extracting DBFSI data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;
  ids          = (int   *) malloc(num*sizeof(int));
  pres         = (float *) malloc(num*sizeof(float));
  fx           = (float *) malloc(num*sizeof(float));
  fy           = (float *) malloc(num*sizeof(float));
  fz           = (float *) malloc(num*sizeof(float));
  pleak        = (float *) malloc(num*sizeof(float));
  flux         = (float *) malloc(num*sizeof(float));
  gx           = (float *) malloc(num*sizeof(float));
  gy           = (float *) malloc(num*sizeof(float));
  gz           = (float *) malloc(num*sizeof(float));
  Ptmp         = (float *) malloc(num*sizeof(float));
  PDt          = (float *) malloc(num*sizeof(float));

  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%sdbfsi",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;

  fprintf(fp,"     Fluid-structure interaction output\n");
  fprintf(fp,"     Number of surfaces:%10d\n\n",num);
  fprintf(fp,"        id          p            fx            fy            fz             mout\n                obsolete         fx-lc         fy-lc         fz-lc         Ptemp         PDtmp\n");
  for(state=1; (dp = next_dir(handle,"/dbfsi",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"fx",0,num,fx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"fy",0,num,fy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"fz",0,num,fz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"pres",0,num,pres) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"mout",0,num,pleak) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"obsolete",0,num,flux) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"gx",0,num,gx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"gy",0,num,gy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"gz",0,num,gz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"Ptmp",0,num,Ptmp) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"PDt",0,num,PDt) != num) break;

    fprintf(fp,"         time=%13.5E\n",time);
    for (i=0;i<num;i++) {
      fprintf(fp,"%10d%14.4E%14.4E%14.4E%14.4E%14.4E\n",ids[i],pres[i],fx[i],fy[i],fz[i],pleak[i]);
      fprintf(fp,"          %14.4E%14.4E%14.4E%14.4E%14.4E%14.4E\n\n",flux[i],gx[i],gy[i],gz[i],Ptmp[i],PDt[i]);
    }
  }
  fclose(fp);

  free(PDt);
  free(Ptmp);
  free(gz);
  free(gy);
  free(gx);
  free(flux);
  free(pleak);
  free(fz);
  free(fy);
  free(fx);
  free(pres);
  free(ids);

  printf("      %d states extracted\n",state-1);
  return 0;
}

static char *tochar2(float value,int len)
{
  static char s[20];

  db_floating_format(value, len, s, 0, len-1);
  return s;
}


   static void db_floating_format(float a, int fw, char *output, int tzero, int nsigs)
/* ==============================
**
** Takes floating point value <a> and renders it "nicely" in a field width
** of <fw>, returning the output right-justified as a character string in
** <output>.  The last character + 1 will be a zero termination.
**
** "Nicely" means it does its best to utilise the field width to get the
** best precision out of the number, but culls any silly looking trailing
** zeros.  It also tries to eliminate any spurious "noise" values by limiting
** mantissa digits to the precision of the incoming argument.
**
** The reason for this distinction is that we can sometimes need to restrict
** the fractional precision of the mantissa of numbers to avoid writing out
** extra, spurious digits that give noise.
**
** If <tzero> is true then trailing zeros are permitted.
**
** <nsigs> is max number of mantissa sig figures, usually 7 or 8.
**
** Field width should not exceed 80, and values less than 8 will probably
** give silly results.
**
** Modified 23/6/2004 to give slightly improved precision for numbers written
** without an exponent, and also to make numbers less than 0.1xxx be written in
** exponential form, (Previously was < 0.01xxx). This prevents unnecessary
** truncation of precision at field widths > 10.
*/

{
    double  b, c, d;
    int     j, k, l, n, np, mw, nf;
    char    b2[81], form[20], *p;


/* Zero is a special case */

    if(a == 0.0)
    {
        for(j=0; j<fw-3; j++) output[j] = ' ';
        strcpy(output+fw-3, "0.0");
        return;
    }


/* Get magnitude power (ie exponent) of incoming value as an integer */

    j = log10(fabs(a)) + 0.0000000001;


/* Use an exponent if the value lies outside the range e7 : e-1,
** but we need exponential format if this is too wide for the
** data field. */

    if(fw < 10) 	/* Small field limits precision */
    {
        if(a < 0.0) np = 5;
        else        np = 6;
    }
    else
    {
        np = 7;
    }

make_exp:

    if(j > np || j < 0)		/* Try for an exponential format */
    {
        if(j < 0) --j;			/* Puts a digit before decimal point */
        c = j;
        d = pow(10.0, c);		/* Scale factor of exponent */
        c = a / d;			/* Factored |mantissa| */
        n = 1;				/* To flag exponent size calc */
    }
    else				/* Non-exponential case */
    {
        c = a;
        d = 1.0;
        n = 0;
    }


loop:

/* Work out how many characters the exponent will require */

    if(n)
    {
             if(j >  99)  n = 4;	/* #characters for exponent */
        else if(j >   9)  n = 3;
        else if(j >  -1)  n = 2;
        else if(j > -10)  n = 3;
        else if(j > -100) n = 4;
        else              n = 5;
    }


/* Mantissa width depends on field width, less any exponent */

    mw = fw - n;


/* Number of fractional places depends on magnitude, precision, and whether we
** need to add a "-" sign. */

    nf = mw - 2;		/* Space for leading digit and decimal point. */
    if(a < 0.0) --nf;		/* Space for "-" sign. */


/* Apply effect of rounding at this fractional precision, and recalculate
** exponent size if round changes order of magnitude. */

    if(n)
    {
        b = -nf - 1;
        b = pow(10.0, b) * 5.0;

        if(fabs(c)+b >= 10.0)
        {
            c /= 10.0;
            j += 1;
            goto loop;
        }
    }


/* Experience on different hardware shows that the 8th sig fig may vary, I presume
** because of rounding/truncation differences between single and double precision
** variables on the different machines.  Trial and error shows that multiplying
** by the following factor reconciles these differences - not a lot of science
** behind this, but it has the merit of working and means that the "common" format
** can give identical output on different hardware!  CB 9/9/2004 */

    c *= 1.0000000001;		/* Correct for rounding/truncation hardware differences */


/* We may need to restrict the resolution of the mantissa fraction to avoid
** getting nunerical noise from attempting to output significant figures for
** which we don't have sufficient data resolution.
**
** The detailed strategy varies because while the exponential case has fixed
** #decimal places, the "fixed" mantissa #decimal places varies.
**
** Note that we could write "if(!n) k = fabs(c) < 1.0 ? j : j+1;" below if we
** wished to eliminate the possibly spurious 8th sig fig. */

    if(!n) k = fabs(c) < 1.0 ? j-1 : j;		/* Allow for value before point */
    else   k = 0;

/* Permit variable #sig figs */

/*  if(nf > 7 - k) nf = 7 - k; */

    if(nf > (l = nsigs - 1 - k)) nf = l;



/* Non-exponential case, not enough space for decimal places */

    if(!n)
    {
        j = fw - k - 1;		/* Space for digits before point, + point itself */
        if(a < 0.0) --j;	/* Space for minus sign */
        if(nf > j) nf = j;	/* Truncate decimal places if necessary */
        if(nf < 0) nf = 0;
    }

/* Write mantissa format statement, and hence the mantissa itself */

prec:

    sprintf(form, "%%#%d.%df", mw, nf);
    sprintf(b2, form, c);



/* Check for non-exponential overflow */

    if(!n && !nf && b2[mw])
    {
        j = log10(fabs(a)) + 0.0000000001;

        if(j < 0) --j;                  /* Puts a digit before decimal point */
        c = j;
        d = pow(10.0, c);               /* Scale factor of exponent */
        c = a / d;                      /* Factored |mantissa| */
        n = 1;                          /* To flag exponent size calc */

        goto loop;
    }

/* And for rounding up "looks odd" error in exponential cases (ie 10.0e-5 instead of 1.0e-4) */

    else if(n)
    {
        if(strstr(b2, "10."))
        {
            c = ++j;
            d = pow(10.0, c);               /* Scale factor of exponent */
            c = a / d;                      /* Factored |mantissa| */
            n = 1;                          /* To flag exponent size calc */

            goto loop;
        }
    }


/* Remove any trailing zeros ... */

    if(!tzero)
    {
        k = 0;
        p = b2+mw-1;
        while (*p == '0')
        {
            --p;
            ++k;
        }

/* ... but if we've arrived at a decimal point then permit one zero. */

        if(*p == '.') --k;


/* Shift string to the right to permit subsequent overwrite of trailing zeros,
** filling vacated leading bytes with spaces. */

        if(k > 0)
        {
            memmove(b2+k, b2, mw);		/* Can copy overlapping objects */
            p = b2;
            while(k--) *p++ = ' ';
        }

/* Otherwise check for possible spurious 8th sig fig. Originally I thought that
** this could be limited to ...999 and ...001, but it seems that ...998 is
** possible (try value 0.9 in single precision), so I assume ...002 is too.
** Therefore I'm now looking at ...00* and ...99* where "*" is in round up/down
** territory as appropriate. */

        else if(nf > 1)
        {
            p = b2+mw-3;

            if((strncmp(p, "00", 2) == 0 && *(p+2) < '5') ||
               (strncmp(p, "99", 2) == 0 && *(p+2) > '5'))
            {
                --nf;
                goto prec;
            }
        }

/* Check for "xxxxxxx00." too, which is better expressed exponentially since we
** can afford to lose the precision */

        else if(nf == 0)
        {
            p = b2+mw-3;

            if(strcmp(p, "00.") == 0)
            {
                np = -1;
                goto make_exp;
            }
        }
    }


/* Add the exponent if required, or just zero terminate string if no exponent */

    switch(n)
    {
        case 0: *(b2+mw) = '\0';             break;
        case 2: sprintf(b2+mw, "E%1d", j);   break;
        case 3: sprintf(b2+mw, "E%2d", j);   break;
        case 4: sprintf(b2+mw, "E%3d", j);   break;
    }


/* Copy into output string (include terminating zero) */

    memcpy(output, b2, fw+1);
}
/*
  ELOUT_SSD file
*/


int translate_elout_ssd(int handle)
{
  int i,j,k,l,m,n,typid,filenum,state;
  LSDA_Length length;
  int have_solid, have_tshell, have_beam, have_shell;
  FILE *fp;

  char path[30];
  int nfreq;
  int h_num , h_nfreq, h_ncomp, h_lnum;
  int t_num , t_nfreq, t_ncomp, t_maxint, t_lnum;
  int b_num , b_nfreq, b_ncomp, b_lnum;
  int s_num , s_nfreq, s_ncomp, s_maxint, s_lnum;
  int *h_uid;
  int *t_uid;
  int *b_uid;
  int *s_uid;
  float h_freq, *h_angle, *h_ampil;
  float t_freq, *t_angle, *t_ampil;
  float b_freq, *b_angle, *b_ampil;
  float s_freq, *s_angle, *s_ampil;
  char title_location[128];

  if (lsda_cd(handle,"/elout_ssd") == -1) return 0;
  printf("Extracting ELOUT_SSD data\n");

  lsda_queryvar(handle,"/elout_ssd/solid",&typid,&length,&filenum);
  have_solid= (typid >= 0);
  lsda_queryvar(handle,"/elout_ssd/thickshell",&typid,&length,&filenum);
  have_tshell= (typid >= 0);
  lsda_queryvar(handle,"/elout_ssd/beam",&typid,&length,&filenum);
  have_beam= (typid >= 0);
  lsda_queryvar(handle,"/elout_ssd/shell",&typid,&length,&filenum);
  have_shell= (typid >= 0);

  title_location[0]=0;
/*
  Read metadata
  Solids
*/
  if(have_solid) {
    lsda_cd(handle,"/elout_ssd/solid/metadata");
    strcpy(title_location,"/elout_ssd/solid/metadata");
    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&h_num);
    lsda_read(handle,LSDA_INT,"nfreq_ssd",0,1,&h_nfreq);
    lsda_read(handle,LSDA_INT,"ncomp",    0,1,&h_ncomp);
    nfreq = h_nfreq;

    h_uid = (int *) malloc(h_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,h_num,h_uid);

    h_lnum = h_ncomp*h_num;
    h_ampil = (float *) malloc(h_lnum*sizeof(float));
    h_angle = (float *) malloc(h_lnum*sizeof(float));
  }
/*
  thick shells
*/
  if(have_tshell) {
    lsda_cd(handle,"/elout_ssd/thickshell/metadata");
    strcpy(title_location,"/elout_ssd/thickshell/metadata");
    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&t_num);
    lsda_read(handle,LSDA_INT,"nfreq_ssd",0,1,&t_nfreq);
    lsda_read(handle,LSDA_INT,"ncomp",    0,1,&t_ncomp);
    lsda_read(handle,LSDA_INT,"maxint",   0,1,&t_maxint);
    nfreq = t_nfreq;

    t_uid = (int *) malloc(t_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,t_num,t_uid);

    t_lnum = t_ncomp*t_num;
    t_ampil = (float *) malloc(t_lnum*sizeof(float));
    t_angle = (float *) malloc(t_lnum*sizeof(float));
  }
/*
  beams
*/
  if(have_beam) {
    lsda_cd(handle,"/elout_ssd/beam/metadata");
    strcpy(title_location,"/elout_ssd/beam/metadata");
    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&b_num);
    lsda_read(handle,LSDA_INT,"nfreq_ssd",0,1,&b_nfreq);
    lsda_read(handle,LSDA_INT,"ncomp",    0,1,&b_ncomp);
    nfreq = b_nfreq;

    b_uid = (int *) malloc(b_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,b_num,b_uid);

    b_lnum = b_ncomp*b_num;
    b_ampil = (float *) malloc(b_lnum*sizeof(float));
    b_angle = (float *) malloc(b_lnum*sizeof(float));
  }
/*
  shells
*/
  if(have_shell) {
    lsda_cd(handle,"/elout_ssd/shell/metadata");
    strcpy(title_location,"/elout_ssd/shell/metadata");
    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&s_num);
    lsda_read(handle,LSDA_INT,"nfreq_ssd",0,1,&s_nfreq);
    lsda_read(handle,LSDA_INT,"ncomp",    0,1,&s_ncomp);
    lsda_read(handle,LSDA_INT,"maxint",   0,1,&s_maxint);
    nfreq = s_nfreq;

    s_uid = (int *) malloc(s_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,s_num,s_uid);

    s_lnum = s_ncomp*s_num;
    s_ampil = (float *) malloc(s_lnum*sizeof(float));
    s_angle = (float *) malloc(s_lnum*sizeof(float));
  }
  if(strlen(title_location) == 0) return 0;
/*
  open file and write header
*/
  sprintf(output_file,"%selout_ssd",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  output_legend(handle,fp,1,1);
  fprintf(fp," S t e a d y   S t a t e   D y n a m i c s\n\n");
/*
  Loop through frequency states and write each one
*/
  for(state=1;(have_solid || have_tshell || have_beam || have_shell) && state<=nfreq ; state++) {

    if(have_solid) {
      if(state<=999999)
        sprintf(path,"/elout_ssd/solid/d%6.6d",state);
      else
        sprintf(path,"/elout_ssd/solid/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&h_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"amplitude",0,h_lnum,h_ampil) != h_lnum) break;
      if(lsda_read(handle,LSDA_FLOAT,"angle",0,h_lnum,h_angle) != h_lnum) break;

      fprintf(fp,"\n element stress calculations for frequency = %12.5E\n\n",h_freq);
      fprintf(fp,"       solid           sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx\n");
      for (m=0; m<h_num; m++) {
        fprintf(fp,"%10d-\n",h_uid[m]);
        l=h_ncomp*m;
        fprintf(fp,"      amplitude  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",h_ampil[l],h_ampil[l+1],h_ampil[l+2],h_ampil[l+3],h_ampil[l+4],h_ampil[l+5]);
        fprintf(fp,"      angle      %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",h_angle[l],h_angle[l+1],h_angle[l+2],h_angle[l+3],h_angle[l+4],h_angle[l+5]);
      }

      if (h_ncomp>7) {
        fprintf(fp,"\n");
        fprintf(fp,"       solid           eps-xx       eps-yy       eps-");
        fprintf(fp,"zz       eps-xy       eps-yz       eps-zx\n");
        for (m=0; m<h_num; m++) {
          fprintf(fp,"%10d-\n",h_uid[m]);
          l=h_ncomp*m+7;
          fprintf(fp,"      amplitude  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",h_ampil[l],h_ampil[l+1],h_ampil[l+2],h_ampil[l+3],h_ampil[l+4],h_ampil[l+5]);
          fprintf(fp,"      angle      %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",h_angle[l],h_angle[l+1],h_angle[l+2],h_angle[l+3],h_angle[l+4],h_angle[l+5]);
        }
      }
    }
    if(have_tshell) {
      if(state<=999999)
        sprintf(path,"/elout_ssd/thickshell/d%6.6d",state);
      else
        sprintf(path,"/elout_ssd/thickshell/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&t_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"amplitude",0,t_lnum,t_ampil) != t_lnum) break;
      if(lsda_read(handle,LSDA_FLOAT,"angle",0,t_lnum,t_angle) != t_lnum) break;

      fprintf(fp,"\n element stress calculations for frequency = %12.5E\n\n",t_freq);
      fprintf(fp," ipt-  thickshl        sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx\n");
      for (m=0; m<t_num; m++) {
        fprintf(fp,"%10d-\n",t_uid[m]);
        for(n=0;n<t_maxint;n++) {
        l=t_ncomp*m + 7*n;
        fprintf(fp,"%4d- amplitude  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",n+1,t_ampil[l],t_ampil[l+1],t_ampil[l+2],t_ampil[l+3],t_ampil[l+4],t_ampil[l+5]);
        fprintf(fp,"%4d- angle      %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",n+1,t_angle[l],t_angle[l+1],t_angle[l+2],t_angle[l+3],t_angle[l+4],t_angle[l+5]);
        }
      }

      if (t_ncomp>7*t_maxint) {
        fprintf(fp,"\n");
        fprintf(fp," ipt-  thickshl        eps-xx       eps-yy       eps-");
        fprintf(fp,"zz       eps-xy       eps-yz       eps-zx\n");
        for (m=0; m<t_num; m++) {
          fprintf(fp,"%10d-\n",t_uid[m]);
          l=t_ncomp*m + 7*t_maxint;
          fprintf(fp," lower ipt- amp  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",t_ampil[l],t_ampil[l+1],t_ampil[l+2],t_ampil[l+3],t_ampil[l+4],t_ampil[l+5]);
          fprintf(fp," lower ipt- ang  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",t_angle[l],t_angle[l+1],t_angle[l+2],t_angle[l+3],t_angle[l+4],t_angle[l+5]);
          l=t_ncomp*m + 7*t_maxint+6;
          fprintf(fp," upper ipt- amp  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",t_ampil[l],t_ampil[l+1],t_ampil[l+2],t_ampil[l+3],t_ampil[l+4],t_ampil[l+5]);
          fprintf(fp," upper ipt- ang  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",t_angle[l],t_angle[l+1],t_angle[l+2],t_angle[l+3],t_angle[l+4],t_angle[l+5]);
        }
      }
    }
    if(have_beam) {
      if(state<=999999)
        sprintf(path,"/elout_ssd/beam/d%6.6d",state);
      else
        sprintf(path,"/elout_ssd/beam/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&b_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"amplitude",0,b_lnum,b_ampil) != b_lnum) break;
      if(lsda_read(handle,LSDA_FLOAT,"angle",0,b_lnum,b_angle) != b_lnum) break;

      fprintf(fp,"\n\n element stress calculations for frequency = %12.5E\n\n",b_freq);
      fprintf(fp,"     resultants         axial      shear-s      she");
      fprintf(fp,"ar-t     moment-s     moment-t      torsion\n");
      for (m=0; m<b_num; m++) {
        fprintf(fp,"%10d-\n",b_uid[m]);
        l=b_ncomp*m;
        fprintf(fp,"      amplitude  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",b_ampil[l],b_ampil[l+1],b_ampil[l+2],b_ampil[l+3],b_ampil[l+4],b_ampil[l+5]);
        fprintf(fp,"      angle      %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",b_angle[l],b_angle[l+1],b_angle[l+2],b_angle[l+3],b_angle[l+4],b_angle[l+5]);
      }

      if (b_ncomp>6) {
        fprintf(fp,"\n     inte. points    ");
        fprintf(fp,"sigma 11     sigma 12     sigma 31    plast eps   axial strn\n");
        for (m=0; m<b_num; m++) {
          fprintf(fp,"%10d-\n",b_uid[m]);
          l=b_ncomp*m+6;
          fprintf(fp,"      amplitude  %12.4E %12.4E %12.4E %12.4E %12.4E\n",b_ampil[l],b_ampil[l+1],b_ampil[l+2],b_ampil[l+3],b_ampil[l+4]);
          fprintf(fp,"      angle      %12.4E %12.4E %12.4E %12.4E %12.4E\n",b_angle[l],b_angle[l+1],b_angle[l+2],b_angle[l+3],b_angle[l+4]);
        }
      }
    }
    if(have_shell) {
      if(state<=999999)
        sprintf(path,"/elout_ssd/shell/d%6.6d",state);
      else
        sprintf(path,"/elout_ssd/shell/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&s_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"amplitude",0,s_lnum,s_ampil) != s_lnum) break;
      if(lsda_read(handle,LSDA_FLOAT,"angle",0,s_lnum,s_angle) != s_lnum) break;

      fprintf(fp,"\n element stress calculations for frequency = %12.5E\n\n",s_freq);
      fprintf(fp," ipt-  shl             sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx\n");
      for (m=0; m<s_num; m++) {
        fprintf(fp,"%10d-\n",s_uid[m]);
        for(n=0;n<s_maxint;n++) {
        l=s_ncomp*m+ 7*n;
        fprintf(fp,"%4d- amplitude  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",n+1,s_ampil[l],s_ampil[l+1],s_ampil[l+2],s_ampil[l+3],s_ampil[l+4],s_ampil[l+5]);
        fprintf(fp,"%4d- angle      %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",n+1,s_angle[l],s_angle[l+1],s_angle[l+2],s_angle[l+3],s_angle[l+4],s_angle[l+5]);
        }
      }

      if (s_ncomp>33){
        fprintf(fp,"\n");
        fprintf(fp," ipt-  shl             eps-xx       eps-yy       eps-");
        fprintf(fp,"zz       eps-xy       eps-yz       eps-zx\n");
        for (m=0; m<s_num; m++) {
          fprintf(fp,"%10d-\n",s_uid[m]);
          l=s_ncomp*m+ 32;
          fprintf(fp," lower ipt- amp  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",s_ampil[l],s_ampil[l+1],s_ampil[l+2],s_ampil[l+3],s_ampil[l+4],s_ampil[l+5]);
          fprintf(fp," lower ipt- ang  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",s_angle[l],s_angle[l+1],s_angle[l+2],s_angle[l+3],s_angle[l+4],s_angle[l+5]);
          l=s_ncomp*m+ 38;
          fprintf(fp," upper ipt- amp  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",s_ampil[l],s_ampil[l+1],s_ampil[l+2],s_ampil[l+3],s_ampil[l+4],s_ampil[l+5]);
          fprintf(fp," upper ipt- ang  %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",s_angle[l],s_angle[l+1],s_angle[l+2],s_angle[l+3],s_angle[l+4],s_angle[l+5]);
        }
      }

      fprintf(fp,"\n");
    }
  }
  fclose(fp);

  if(have_solid) {
    free (h_angle);
    free (h_ampil);
    free (h_uid);
  }
  if(have_tshell) {
    free (t_angle);
    free (t_ampil);
    free (t_uid);
  }
  if(have_beam) {
    free (b_angle);
    free (b_ampil);
    free (b_uid);
  }
  if(have_shell) {
    free (s_angle);
    free (s_ampil);
    free (s_uid);
  }

  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  ELOUT_SPCM file
*/


int translate_elout_spcm(int handle)
{
  int i,j,k,l,m,n,typid,filenum,state;
  LSDA_Length length;
  int have_solid, have_tshell, have_beam, have_shell;
  FILE *fp;

  char path[30];
  int h_num , h_lnum;
  int t_num , t_maxint, t_lnum;
  int b_num , b_lnum;
  int s_num , s_maxint, s_lnum;
  int *h_uid;
  int *t_uid;
  int *b_uid;
  int *s_uid;
  float *h_strs;
  float *t_strs;
  float *b_strs;
  float *s_strs;
  char title_location[128];

  if (lsda_cd(handle,"/elout_spcm") == -1) return 0;
  printf("Extracting ELOUT_SPCM data\n");

  lsda_queryvar(handle,"/elout_spcm/solid",&typid,&length,&filenum);
  have_solid= (typid >= 0);
  lsda_queryvar(handle,"/elout_spcm/thickshell",&typid,&length,&filenum);
  have_tshell= (typid >= 0);
  lsda_queryvar(handle,"/elout_spcm/beam",&typid,&length,&filenum);
  have_beam= (typid >= 0);
  lsda_queryvar(handle,"/elout_spcm/shell",&typid,&length,&filenum);
  have_shell= (typid >= 0);

  title_location[0]=0;
/*
  Read uid and stress
  Solids
*/
  if(have_solid) {
    lsda_cd(handle,"/elout_spcm/solid");
    strcpy(title_location,"/elout_spcm/solid/metadata");
    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&h_num);

    h_uid = (int *) malloc(h_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,h_num,h_uid);

    h_lnum = 7*h_num;
    h_strs= (float *) malloc(h_lnum*sizeof(float));
    lsda_read(handle,LSDA_FLOAT,"stress",0,h_lnum,h_strs);
    lsda_cd(handle,"/elout_spcm/solid/metadata");
  }
/*
  thick shells
*/
  if(have_tshell) {
    lsda_cd(handle,"/elout_spcm/thickshell");
    strcpy(title_location,"/elout_spcm/thickshell/metadata");

    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&t_num);
    lsda_read(handle,LSDA_INT,"maxint",   0,1,&t_maxint);

    t_uid = (int *) malloc(t_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,t_num,t_uid);

    t_lnum = 7*t_maxint*t_num;
    t_strs= (float *) malloc(t_lnum*sizeof(float));
    lsda_read(handle,LSDA_FLOAT,"stress",0,t_lnum,t_strs);
    lsda_cd(handle,"/elout_spcm/thickshell/metadata");
  }
/*
  beams
*/
  if(have_beam) {
    lsda_cd(handle,"/elout_spcm/beam");
    strcpy(title_location,"/elout_spcm/beam/metadata");

    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&b_num);

    b_uid = (int *) malloc(b_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,b_num,b_uid);

    b_lnum = 7*b_num;
    b_strs= (float *) malloc(b_lnum*sizeof(float));
    lsda_read(handle,LSDA_FLOAT,"stress",0,b_lnum,b_strs);
    lsda_cd(handle,"/elout_spcm/beam/metadata");
  }
/*
  shells
*/
  if(have_shell) {
    lsda_cd(handle,"/elout_spcm/shell");
    strcpy(title_location,"/elout_spcm/shell/metadata");

    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&s_num);
    lsda_read(handle,LSDA_INT,"maxint",   0,1,&s_maxint);

    s_uid = (int *) malloc(s_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,s_num,s_uid);

    s_lnum = 7*s_maxint*s_num;
    s_strs= (float *) malloc(s_lnum*sizeof(float));
    lsda_read(handle,LSDA_FLOAT,"stress",0,s_lnum,s_strs);
    lsda_cd(handle,"/elout_spcm/shell/metadata");
  }
  if(strlen(title_location) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  sprintf(output_file,"%selout_spcm",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  output_legend(handle,fp,1,1);
  fprintf(fp," R e s p o n s e  S p e c t r u m\n\n");
/*
  write stress data
*/
    if(have_solid) {
      sprintf(path,"/elout_spcm/solid");
      path[17]='\0';
      lsda_cd(handle,path);
      fprintf(fp,"\n element stress calculations \n\n");
      fprintf(fp,"       solid      sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx     vmstress\n");
      for (m=0; m<h_num; m++) {
        l=7*m;
        fprintf(fp,"%10d- %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",h_uid[m],h_strs[l],h_strs[l+1],h_strs[l+2],h_strs[l+3],h_strs[l+4],h_strs[l+5],h_strs[l+6]);
      }
    }
    if(have_tshell) {
      sprintf(path,"/elout_spcm/thickshell");
      path[22]='\0';
      lsda_cd(handle,path);
      fprintf(fp,"\n element stress calculations\n\n");
      fprintf(fp," ipt-  thickshl sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx     vmstress\n");
      for (m=0; m<t_num; m++) {
        fprintf(fp,"%10d-\n",t_uid[m]);
        for(n=0;n<t_maxint;n++) {
        l=7*m*t_maxint + 7*n;
        fprintf(fp,"%4d-     %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",n+1,t_strs[l],t_strs[l+1],t_strs[l+2],t_strs[l+3],t_strs[l+4],t_strs[l+5],t_strs[l+6]);
        }
      }
    }
    if(have_beam) {
      sprintf(path,"/elout_spcm/beam");
      path[16]='\0';
      lsda_cd(handle,path);
      fprintf(fp,"\n element stress calculations\n\n");
      fprintf(fp,"       beam       sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx     vmstress\n");
      for (m=0; m<b_num; m++) {
        l=7*m;
        fprintf(fp,"%10d- %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",b_uid[m],b_strs[l],b_strs[l+1],b_strs[l+2],b_strs[l+3],b_strs[l+4],b_strs[l+5],b_strs[l+6]);
      }
    }
    if(have_shell) {
      sprintf(path,"/elout_spcm/shell");
      path[17]='\0';
      lsda_cd(handle,path);
      fprintf(fp,"\n element stress calculations\n\n");
      fprintf(fp," ipt-  shl      sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx     vmstress\n");
      for (m=0; m<s_num; m++) {
        fprintf(fp,"%10d-\n",s_uid[m]);
        for(n=0;n<s_maxint;n++) {
        l=7*m*s_maxint + 7*n;
        fprintf(fp,"%4d-     %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",n+1,s_strs[l],s_strs[l+1],s_strs[l+2],s_strs[l+3],s_strs[l+4],s_strs[l+5],s_strs[l+6]);
        }
      }
      fprintf(fp,"\n");
    }
  fclose(fp);

  if(have_solid) {
    free (h_strs);
    free (h_uid);
  }
  if(have_tshell) {
    free (t_strs);
    free (t_uid);
  }
  if(have_beam) {
    free (b_strs);
    free (b_uid);
  }
  if(have_shell) {
    free (s_strs);
    free (s_uid);
  }

  printf("      %d states extracted\n",1);
  return 0;
}
/*
  ELOUT_PSD file
*/


int translate_elout_psd(int handle)
{
  int i,j,k,l,m,n,typid,filenum,state;
  LSDA_Length length;
  int have_solid, have_tshell, have_beam, have_shell;
  FILE *fp;

  char path[30];
  int nfreq;
  int h_num , h_nfreq, h_lnum;
  int t_num , t_nfreq, t_maxint, t_lnum;
  int b_num , b_nfreq, b_lnum;
  int s_num , s_nfreq, s_maxint, s_lnum;
  int *h_uid;
  int *t_uid;
  int *b_uid;
  int *s_uid;
  float h_freq, *h_strss;
  float t_freq, *t_strss;
  float b_freq, *b_strss;
  float s_freq, *s_strss;
  char title_location[128];

  if (lsda_cd(handle,"/elout_psd") == -1) return 0;
  printf("Extracting ELOUT_PSD data\n");

  lsda_queryvar(handle,"/elout_psd/solid",&typid,&length,&filenum);
  have_solid= (typid >= 0);
  lsda_queryvar(handle,"/elout_psd/thickshell",&typid,&length,&filenum);
  have_tshell= (typid >= 0);
  lsda_queryvar(handle,"/elout_psd/beam",&typid,&length,&filenum);
  have_beam= (typid >= 0);
  lsda_queryvar(handle,"/elout_psd/shell",&typid,&length,&filenum);
  have_shell= (typid >= 0);

  title_location[0]=0;
/*
  Read metadata
  Solids
*/
  if(have_solid) {
    lsda_cd(handle,"/elout_psd/solid/metadata");
    strcpy(title_location,"/elout_psd/solid/metadata");

    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&h_num);
    lsda_read(handle,LSDA_INT,"nfreq_psd",0,1,&h_nfreq);
    nfreq = h_nfreq;

    h_uid = (int *) malloc(h_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,h_num,h_uid);

    h_lnum = 7*h_num;
    h_strss = (float *) malloc(h_lnum*sizeof(float));
  }
/*
  thick shells
*/
  if(have_tshell) {
    lsda_cd(handle,"/elout_psd/thickshell/metadata");
    strcpy(title_location,"/elout_psd/thickshell/metadata");

    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&t_num);
    lsda_read(handle,LSDA_INT,"nfreq_psd",0,1,&t_nfreq);
    lsda_read(handle,LSDA_INT,"maxint",   0,1,&t_maxint);
    nfreq = t_nfreq;

    t_uid = (int *) malloc(t_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,t_num,t_uid);

    t_lnum = 7*t_maxint*t_num;
    t_strss = (float *) malloc(t_lnum*sizeof(float));
  }
/*
  beams
*/
  if(have_beam) {
    lsda_cd(handle,"/elout_psd/beam/metadata");
    strcpy(title_location,"/elout_psd/beam/metadata");

    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&b_num);
    lsda_read(handle,LSDA_INT,"nfreq_psd",0,1,&b_nfreq);
    nfreq = b_nfreq;

    b_uid = (int *) malloc(b_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,b_num,b_uid);

    b_lnum = 7*b_num;
    b_strss = (float *) malloc(b_lnum*sizeof(float));
  }
/*
  shells
*/
  if(have_shell) {
    lsda_cd(handle,"/elout_psd/shell/metadata");
    strcpy(title_location,"/elout_psd/shell/metadata");

    lsda_read(handle,LSDA_INT,"n_elem",   0,1,&s_num);
    lsda_read(handle,LSDA_INT,"nfreq_psd",0,1,&s_nfreq);
    lsda_read(handle,LSDA_INT,"maxint",   0,1,&s_maxint);
    nfreq = s_nfreq;

    s_uid = (int *) malloc(s_num*sizeof(int));
    lsda_read(handle,LSDA_INT,"uid",0,s_num,s_uid);

    s_lnum = 7*s_maxint*s_num;
    s_strss = (float *) malloc(s_lnum*sizeof(float));
  }
  if(strlen(title_location) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  sprintf(output_file,"%selout_psd",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  output_legend(handle,fp,1,1);
  fprintf(fp," R a n d o m   V i b r a t i o n\n\n");
/*
  Loop through frequency states and write each one
*/
  for(state=1;(have_solid || have_tshell || have_beam || have_shell) && state<=nfreq ; state++) {

    if(have_solid) {
      if(state<=999999)
        sprintf(path,"/elout_psd/solid/d%6.6d",state);
      else
        sprintf(path,"/elout_psd/solid/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&h_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"stress",0,h_lnum,h_strss) != h_lnum) break;
      fprintf(fp,"\n element stress calculations for frequency = %12.5E\n\n",h_freq);
      fprintf(fp,"       solid           sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx       sig-vm\n");
      for (m=0; m<h_num; m++) {
        l=7*m;
        fprintf(fp,"%10d-      %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",h_uid[m],h_strss[l],h_strss[l+1],h_strss[l+2],h_strss[l+3],h_strss[l+4],h_strss[l+5],h_strss[l+6]);
      }
    }
    if(have_tshell) {
      if(state<=999999)
        sprintf(path,"/elout_psd/thickshell/d%6.6d",state);
      else
        sprintf(path,"/elout_psd/thickshell/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&t_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"stress",0,t_lnum,t_strss) != t_lnum) break;
      fprintf(fp,"\n element stress calculations for frequency = %12.5E\n\n",t_freq);
      fprintf(fp,"  thickshl  ipt-       sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx       sig-vm\n");
      for (m=0; m<t_num; m++) {
        for(n=0;n<t_maxint;n++) {
        l=7*m*t_maxint + 7*n;
        fprintf(fp,"%10d-%4d- %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",t_uid[m],n+1,t_strss[l],t_strss[l+1],t_strss[l+2],t_strss[l+3],t_strss[l+4],t_strss[l+5],t_strss[l+6]);
        }
      }
    }
    if(have_beam) {
      if(state<=999999)
        sprintf(path,"/elout_psd/beam/d%6.6d",state);
      else
        sprintf(path,"/elout_psd/beam/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&b_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"stress",0,b_lnum,b_strss) != b_lnum) break;
      fprintf(fp,"\n element stress calculations for frequency = %12.5E\n\n",b_freq);
      fprintf(fp,"       beam            sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx       sig-vm\n");
      for (m=0; m<b_num; m++) {
        l=7*m;
        fprintf(fp,"%10d-      %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",b_uid[m],b_strss[l],b_strss[l+1],b_strss[l+2],b_strss[l+3],b_strss[l+4],b_strss[l+5],b_strss[l+6]);
      }
    }
    if(have_shell) {
      if(state<=999999)
        sprintf(path,"/elout_psd/shell/d%6.6d",state);
      else
        sprintf(path,"/elout_psd/shell/d%8.8d",state);
      lsda_cd(handle,path);
      if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&s_freq) != 1) break;
      if(lsda_read(handle,LSDA_FLOAT,"stress",0,s_lnum,s_strss) != s_lnum) break;
      fprintf(fp,"\n element stress calculations for frequency = %12.5E\n\n",s_freq);
      fprintf(fp,"       shl  ipt-       sig-xx       sig-yy       sig-");
      fprintf(fp,"zz       sig-xy       sig-yz       sig-zx       sig-vm\n");
      for (m=0; m<s_num; m++) {
        for(n=0;n<s_maxint;n++) {
        l=7*m*s_maxint + 7*n;
        fprintf(fp,"%10d-%4d- %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E %12.4E\n",s_uid[m],n+1,s_strss[l],s_strss[l+1],s_strss[l+2],s_strss[l+3],s_strss[l+4],s_strss[l+5],s_strss[l+6]);
        }
      }
      fprintf(fp,"\n");
    }
  }
  fclose(fp);

  if(have_solid) {
    free (h_strss);
    free (h_uid);
  }
  if(have_tshell) {
    free (t_strss);
    free (t_uid);
  }
  if(have_beam) {
    free (b_strss);
    free (b_uid);
  }
  if(have_shell) {
    free (s_strss);
    free (s_uid);
  }

  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  CURVOUT file
*/
int translate_curvout(int handle)
{
  int i,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[128];
  int *ids;
  int cycle;
  float time;
  float *v;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/curvout/metadata") == -1) return 0;
  printf("Extracting CURVOUT data\n");

  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num=length;

/*
  allocate memory to read in 1 state
*/
  ids = (int *) malloc(num*sizeof(int));
  v   = (float *) malloc(num*sizeof(float));
/*
  Read metadata
*/
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%scurvout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,dirname,fp);
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/curvout",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"values",0,num,v) != num) break;

    fprintf(fp,"\n curve data for time step %10d ( at time%12.5E )\n",cycle,time);
    fprintf(fp,"\n curve id    ordinate\n");
    for(i=0; i<num; i++) {
      fprintf(fp,"%9d%13.4E\n",ids[i],v[i]);
    }
  }
  fclose(fp);
  free(v);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  NODOUT_SSD file
*/
int translate_nodout_ssd(int handle)
{
  int i,l,m,typid,num,filenum,state;
  LSDA_Length length;
  FILE *fp;

  char path[30];
  int  nfreq,lnum;
  int *uid;
  float time;
  float freq, *angle,*ampil;
  char title_location[128];

  if (lsda_cd(handle,"/nodout_ssd") == -1) return 0;
  printf("Extracting NODOUT_SSD data\n");

  title_location[0]=0;
/*
  Read metadata
*/
  lsda_cd(handle,"/nodout_ssd/metadata");
  strcpy(title_location,"/nodout_ssd/metadata");
  lsda_read(handle,LSDA_INT,"n_node",   0,1,&num);
  lsda_read(handle,LSDA_INT,"nfreq_ssd",0,1,&nfreq);
/*
  allocate memory to read in 1 state
*/
  uid = (int *) malloc(num*sizeof(int));
  lsda_read(handle,LSDA_INT,"uid",0,num,uid);
  lnum = 9*num;
  ampil = (float *) malloc(lnum*sizeof(float));
  angle = (float *) malloc(lnum*sizeof(float));
  if(strlen(title_location) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  sprintf(output_file,"%snodout_ssd",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  output_legend(handle,fp,1,1);
  fprintf(fp," S t e a d y   S t a t e   D y n a m i c s\n\n");
/*
  Loop through frequency states and write each one
*/
  for(state=1; state<=nfreq; state++) {
    if(state<=999999)
      sprintf(path,"/nodout_ssd/d%6.6d",state);
    else
      sprintf(path,"/nodout_ssd/d%8.8d",state);
    lsda_cd(handle,path);
    if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&freq) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"amplitude",0,lnum,ampil) != lnum) break;
    if(lsda_read(handle,LSDA_FLOAT,"angle",0,lnum,angle) != lnum) break;
    fprintf(fp,"\n nodal print out for frequency = %12.5E\n\n",freq);
    fprintf(fp," nodal point      x-disp      y-disp      z-disp       ");
    fprintf(fp,"x-vel       y-vel       z-vel      x-accl      y-accl      ");
    fprintf(fp,"z-accl\n");
    for (m=0; m<num; m++) {
       fprintf(fp,"%10d-\n",uid[m]);
       l=9*m;
       fprintf(fp," amplitude  %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",ampil[l],ampil[l+3],ampil[l+6],ampil[l+1],ampil[l+4],ampil[l+7],ampil[l+2],ampil[l+5],ampil[l+8]);
       fprintf(fp," angle      %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",angle[l],angle[l+3],angle[l+6],angle[l+1],angle[l+4],angle[l+7],angle[l+2],angle[l+5],angle[l+8]);
      }
  }
  fclose(fp);
  free (uid);
  free (ampil);
  free (angle);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  NODOUT_SPCM file
*/
int translate_nodout_spcm(int handle)
{
  int i,l,m,num,filenum;
  LSDA_Length length;
  FILE *fp;

  char path[30];
  int  lnum;
  int *uid;
  float *vad;
  char title_location[128];

  if (lsda_cd(handle,"/nodout_spcm") == -1) return 0;
  printf("Extracting NODOUT_SPCM data\n");

  title_location[0]=0;
/*
  Read metadata
*/
  lsda_cd(handle,"/nodout_spcm");
  strcpy(title_location,"/nodout_spcm/metadata");
  lsda_read(handle,LSDA_INT,"n_node",   0,1,&num);
/*
  allocate memory to read in 1 state
*/
  uid = (int *) malloc(num*sizeof(int));
  lsda_read(handle,LSDA_INT,"uid",0,num,uid);
  lnum = 9*num;
  vad = (float *) malloc(lnum*sizeof(float));
  lsda_read(handle,LSDA_FLOAT,"vad",0,lnum,vad);
  lsda_cd(handle,"/nodout_spcm/metadata");
  if(strlen(title_location) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  sprintf(output_file,"%snodout_spcm",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  output_legend(handle,fp,1,1);
  fprintf(fp," R e s p o n s e  S p e c t r u m\n\n");
/*
  write results
*/
  sprintf(path,"/nodout_spcm");
  path[12]='\0';
  lsda_cd(handle,path);
  fprintf(fp,"\n nodal print out\n\n");
  fprintf(fp," nodal point      x-disp      y-disp      z-disp       ");
  fprintf(fp,"x-vel       y-vel       z-vel      x-accl      y-accl      ");
  fprintf(fp,"z-accl\n");
  for (m=0; m<num; m++) {
     l=9*m;
     fprintf(fp,"%10d- %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",uid[m],vad[l],vad[l+3],vad[l+6],vad[l+1],vad[l+4],vad[l+7],vad[l+2],vad[l+5],vad[l+8]);
  }
  fclose(fp);
  free (uid);
  free (vad);
  printf("      %d states extracted\n",1);
  return 0;
}
/*
  NODOUT_PSD file
*/
int translate_nodout_psd(int handle)
{
  int i,l,m,typid,num,filenum,state;
  LSDA_Length length;
  FILE *fp;

  char path[30];
  int  nfreq,lnum;
  int *uid;
  float time;
  float freq, *disp,*vel,*acl;
  char title_location[128];

  if (lsda_cd(handle,"/nodout_psd") == -1) return 0;
  printf("Extracting NODOUT_PSD data\n");

  title_location[0]=0;
/*
  Read metadata
*/
  lsda_cd(handle,"/nodout_psd/metadata");
  strcpy(title_location,"/nodout_psd/metadata");
  lsda_read(handle,LSDA_INT,"n_node",   0,1,&num);
  lsda_read(handle,LSDA_INT,"nfreq_psd",0,1,&nfreq);
/*
  allocate memory to read in 1 state
*/
  uid = (int *) malloc(num*sizeof(int));
  lsda_read(handle,LSDA_INT,"uid",0,num,uid);
  lnum = 3*num;
  disp = (float *) malloc(lnum*sizeof(float));
  vel  = (float *) malloc(lnum*sizeof(float));
  acl  = (float *) malloc(lnum*sizeof(float));
  if(strlen(title_location) == 0) return 0;  /* huh? */
/*
  open file and write header
*/
  sprintf(output_file,"%snodout_psd",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,title_location,fp);
  output_legend(handle,fp,1,1);
  fprintf(fp," R a n d o m   V i b r a t i o n\n\n");
/*
  Loop through frequency states and write each one
*/
  for(state=1; state<=nfreq; state++) {
    if(state<=999999)
      sprintf(path,"/nodout_psd/d%6.6d",state);
    else
      sprintf(path,"/nodout_psd/d%8.8d",state);
    lsda_cd(handle,path);
    if(lsda_read(handle,LSDA_FLOAT,"frequency",0,1,&freq) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"displacement",0,lnum,disp) != lnum) break;
    if(lsda_read(handle,LSDA_FLOAT,"velocity",0,lnum,vel) != lnum) break;
    if(lsda_read(handle,LSDA_FLOAT,"acceleration",0,lnum,acl) != lnum) break;
    fprintf(fp,"\n nodal print out for frequency = %12.5E\n\n",freq);
    fprintf(fp," nodal point      x-disp      y-disp      z-disp       ");
    fprintf(fp,"x-vel       y-vel       z-vel      x-accl      y-accl      ");
    fprintf(fp,"z-accl\n");
    for (m=0; m<num; m++) {
       l=3*m;
       fprintf(fp,"%10d- %12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E%12.4E\n",uid[m],disp[l],disp[l+1],disp[l+2],vel[l],vel[l+1],vel[l+2],acl[l],acl[l+1],acl[l+2]);
      }
  }
  fclose(fp);
  free (uid);
  free (disp);
  free (vel);
  free (acl);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  PLLYOUT file
*/
int translate_pllyout(int handle)
{
  int i,typid,num,filenum,state;
  LSDA_Length length;
  int *ids;
  int cycle;
  double time;
  int *b1,*b2;
  float *s,*sr,*f,*wa;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/pllyout/metadata") == -1) return 0;
  printf("Extracting PLLYOUT data\n");

  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num=length;

/*
  allocate memory to read in 1 state
*/
  ids = (int *) malloc(num*sizeof(int));
  b1  = (int *) malloc(num*sizeof(int));
  b2  = (int *) malloc(num*sizeof(int));
  s   = (float *) malloc(num*sizeof(float));
  sr  = (float *) malloc(num*sizeof(float));
  f   = (float *) malloc(num*sizeof(float));
  wa  = (float *) malloc(num*sizeof(float));
/*
  Read metadata
*/
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%spllyout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/pllyout/metadata",fp);
/* no legend so far...  output_legend(handle,fp,1,1); */
/*
  Loop through time states and write each one
*/
  for(state=1; next_dir_6or8digitd(handle,"/pllyout",state) != 0; state++) {
    if(lsda_read(handle,LSDA_DOUBLE,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"slip",0,num,s) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"slip_rate",0,num,sr) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"wrap_angle",0,num,wa) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"force",0,num,f) != num) break;
    if(lsda_read(handle,LSDA_INT,"beam1",0,num,b1) != num) break;
    if(lsda_read(handle,LSDA_INT,"beam2",0,num,b2) != num) break;

    fprintf(fp,"\n\n p u l l e y   e l e m e n t   o u t p u t   ");
    fprintf(fp,"f o r   t i m e   s t e p%9d  ( at time%14.7E )\n",cycle,time);
    fprintf(fp,"\n    pulley #    beam 1 #    beam 2 #          slip");
    fprintf(fp,"      sliprate         force    wrap angle\n");
    for(i=0; i<num; i++) {
      fprintf(fp,"\n%12d%12d%12d%14.5E%14.5E%14.5E%14.5E\n",ids[i],b1[i],b2[i],s[i],
              sr[i],f[i],wa[i]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
  free(wa);
  free(f);
  free(sr);
  free(s);
  free(b2);
  free(b1);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  DEM RCFORC file
*/
int translate_dem_rcforc(int handle)
{
  int i,j,typid,num,filenum,state;
  LSDA_Length length;
  char dirname[32];
  int *ids;
  int *sides, *single, nsingle, nsout;
  float *xf,*yf,*zf;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/dem_rcforc/metadata") == -1) return 0;
  printf("Extracting DEM RCFORC data\n");
/*
  Read metadata
*/
  lsda_queryvar(handle,"ids",&typid,&length,&filenum);
  num = length;

  ids = (int *) malloc(num*sizeof(int));
  xf = (float *) malloc(num*sizeof(float));
  yf = (float *) malloc(num*sizeof(float));
  zf = (float *) malloc(num*sizeof(float));
  
  lsda_read(handle,LSDA_INT,"ids",0,num,ids);
/*
  open file and write header
*/
  sprintf(output_file,"%sdemrcf",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_legend(handle,fp,1,1);
/*
  Loop through time states and write each one.
*/
  for(state=1; (dp = next_dir(handle,"/dem_rcforc",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"x_force",0,num,xf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y_force",0,num,yf) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z_force",0,num,zf) != num) break;
    nsout=0;
    for(i=0; i<num; i++) {
      fprintf(fp,"  \n");
      fprintf(fp,"  master%11d time",ids[i]);
      fprintf(fp,"%12.5E  x %12.5E  y %12.5E  z %12.5E\n",
        time,xf[i],yf[i],zf[i]);
    }
  }
  fclose(fp);
  free(zf);
  free(yf);
  free(xf);
  free(ids);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  DISBOUT file
*/
int translate_disbout(int handle)
{
  int i,typid,num,filenum,state,cycle;
  LSDA_Length length;
  char dirname[256];
  int *nelb, *matid, *mtype;
  float *r_dis_axial,*r_dis_ns,*r_dis_nt,*axial_rot,*rot_s,*rot_t;
  float *rslt_axial,*rslt_ns,*rslt_nt,*torsion,*moment_s,*moment_t;
  float *axial_x,*axial_y,*axial_z,*s_dir_x,*s_dir_y,*s_dir_z,*t_dir_x,*t_dir_y,*t_dir_z;
  float time;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/disbout") == -1) return 0;
  printf("Extracting DISBOUT data\n");
/*
  Read metadata
*/

/*
  open file and write header
*/
  sprintf(output_file,"%sdisbout",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;
  output_title(handle,"/disbout/metadata",fp);
  output_legend(handle,fp,1,1);

  for(state=1; (dp = next_dir(handle,"/disbout",dp,dirname)) != NULL; state++) {
  if(state<=999999)
    sprintf(dirname,"/disbout/d%6.6d",state);
  else
    sprintf(dirname,"/disbout/d%8.8d",state);

  lsda_queryvar(handle,dirname,&typid,&length,&filenum);
/*  if(typid != 0) return 0; */
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"nelb",&typid,&length,&filenum);
  num = length;

  nelb  = (int *) malloc(num*sizeof(int));
  matid = (int *) malloc(num*sizeof(int));
  mtype = (int *) malloc(num*sizeof(int));
  r_dis_axial = (float *) malloc(num*sizeof(float));
  r_dis_ns    = (float *) malloc(num*sizeof(float));
  r_dis_nt    = (float *) malloc(num*sizeof(float));
  axial_rot   = (float *) malloc(num*sizeof(float));
  rot_s       = (float *) malloc(num*sizeof(float));
  rot_t       = (float *) malloc(num*sizeof(float));
  rslt_axial = (float *) malloc(num*sizeof(float));
  rslt_ns    = (float *) malloc(num*sizeof(float));
  rslt_nt    = (float *) malloc(num*sizeof(float));
  torsion    = (float *) malloc(num*sizeof(float));
  moment_s   = (float *) malloc(num*sizeof(float));
  moment_t   = (float *) malloc(num*sizeof(float));
  axial_x = (float *) malloc(num*sizeof(float));
  axial_y = (float *) malloc(num*sizeof(float));
  axial_z = (float *) malloc(num*sizeof(float));
  s_dir_x = (float *) malloc(num*sizeof(float));
  s_dir_y = (float *) malloc(num*sizeof(float));
  s_dir_z = (float *) malloc(num*sizeof(float));
  t_dir_x = (float *) malloc(num*sizeof(float));
  t_dir_y = (float *) malloc(num*sizeof(float));
  t_dir_z = (float *) malloc(num*sizeof(float));
/*
  Loop through time states and write each one.
*/
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"cycle",0,1,&cycle) != 1) break;
    if(lsda_read(handle,LSDA_INT,"nelb",0,num,nelb) != num) break;
    if(lsda_read(handle,LSDA_INT,"matid",0,num,matid) != num) break;
    if(lsda_read(handle,LSDA_INT,"mtype",0,num,mtype) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"r_dis_axial",0,num,r_dis_axial) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"r_dis_ns",0,num,r_dis_ns) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"r_dis_nt",0,num,r_dis_nt) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"axial_rot",0,num,axial_rot) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rot_s",0,num,rot_s) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rot_t",0,num,rot_t) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rslt_axial",0,num,rslt_axial) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rslt_ns",0,num,rslt_ns) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rslt_nt",0,num,rslt_nt) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"torsion",0,num,torsion) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"moment_s",0,num,moment_s) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"moment_t",0,num,moment_t) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"axial_x",0,num,axial_x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"axial_y",0,num,axial_y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"axial_z",0,num,axial_z) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"s_dir_x",0,num,s_dir_x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"s_dir_y",0,num,s_dir_y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"s_dir_z",0,num,s_dir_z) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"t_dir_x",0,num,t_dir_x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"t_dir_y",0,num,t_dir_y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"t_dir_z",0,num,t_dir_z) != num) break;

    fprintf(fp,"\n d i s c r e t e  b e a m  o u t p u t  f o r   t i m e  s t e p");
      fprintf(fp,"%9d   ( at time%12.5E )\n\n",cycle,time);

    for(i=0; i<num; i++) {
      fprintf(fp," beam/truss # =%8d      part ID  =%10d      material type=%8d\n",nelb[i],matid[i],mtype[i]);
      fprintf(fp," relative displ. axial   normal-s   normal-t  axial rot rotation-s rotation-t\n");
       fprintf(fp,"           %11.3E%11.3E%11.3E%11.3E%11.3E%11.3E\n",
         r_dis_axial[i],r_dis_ns[i],r_dis_nt[i],axial_rot[i],rot_s[i],rot_t[i]);
      fprintf(fp," local resultant axial   normal-s   normal-t    torsion   moment-s  moment-t\n");
       fprintf(fp,"           %11.3E%11.3E%11.3E%11.3E%11.3E%11.3E\n",
         rslt_axial[i],rslt_ns[i],rslt_nt[i],torsion[i],moment_s[i],moment_t[i]);
      fprintf(fp," axial direction     x          y          z\n");
       fprintf(fp,"           %11.3E%11.3E%11.3E\n",axial_x[i],axial_y[i],axial_z[i]);
      fprintf(fp," s direction         x          y          z\n");
       fprintf(fp,"           %11.3E%11.3E%11.3E\n",s_dir_x[i],s_dir_y[i],s_dir_z[i]);
      fprintf(fp," t direction         x          y          z\n");
       fprintf(fp,"           %11.3E%11.3E%11.3E\n\n",t_dir_x[i],t_dir_y[i],t_dir_z[i]);
      }

    }
  fclose(fp);
  free(nelb);
  free(matid);
  free(mtype);
  free(r_dis_axial);
  free(r_dis_ns);
  free(r_dis_nt);
  free(axial_rot);
  free(rot_s);
  free(rot_t);
  free(rslt_axial);
  free(rslt_ns);
  free(rslt_nt);
  free(torsion);
  free(moment_s);
  free(moment_t);
  free(axial_x);
  free(axial_y);
  free(axial_z);
  free(s_dir_x);
  free(s_dir_y);
  free(s_dir_z);
  free(t_dir_x);
  free(t_dir_y);
  free(t_dir_z);
  printf("      %d states extracted\n",state-1);
  return 0;
}
/*
  DEM TRHIST file
*/
int translate_dem_trhist(int handle)
{
  int i,typid,num,num0,filenum,state;
  LSDA_Length length;
  char dirname[128];
  int maxnode,minnode,nstep,iteration;
  float maxtemp,mintemp,*temp,temp_norm;
  float time,timestep,*t_bottom,*t_top;
  float *fiop,*x,*y,*z,*vx,*vy,*vz,*sx,*sy,*sz,*sxy,*syz,*szx,*efp,*rvl,*rho;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/demtrh/metadata") == -1) return 0;
  printf("Extracting TRHIST data\n");

  sprintf(dirname,"/demtrh/d000001");
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"fiop",&typid,&length,&filenum);
  num=length;
/*
  allocate memory to read in 1 state
*/
  fiop   = (float *) malloc(num*sizeof(float));
  x      = (float *) malloc(num*sizeof(float));
  y      = (float *) malloc(num*sizeof(float));
  z      = (float *) malloc(num*sizeof(float));
  vx     = (float *) malloc(num*sizeof(float));
  vy     = (float *) malloc(num*sizeof(float));
  vz     = (float *) malloc(num*sizeof(float));
/*
  open file and write header
*/
  sprintf(output_file,"%sdemtrh",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;

  fprintf(fp,"DEM Tracer particle file\n");
  fprintf(fp,"%5d   16\n",num);
  fprintf(fp,"            x            y            z           vx           vy           vz\n");
/*
  output_title(handle,"/trhist/metadata",fp);
  output_legend(handle,fp,1,1);
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/demtrh",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_FLOAT,"fiop",0,num,fiop) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x",0,num,x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y",0,num,y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z",0,num,z) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vx",0,num,vx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vy",0,num,vy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vz",0,num,vz) != num) break;

    fprintf(fp,"%13.5E\n",time);
    for (i=0;i<num;i++) {
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",x[i],y[i],z[i],vx[i],vy[i],vz[i]);
    }
  }
  fclose(fp);

  free(vz);
  free(vy);
  free(vx);
  free(z);
  free(y);
  free(x);
  free(fiop);

  printf("      %d states extracted\n",state-1);
  return 0;
}

/*
  TRALEH file
*/
int translate_traleh(int handle)
{
  int i,typid,num,num0,filenum,state;
  LSDA_Length length;
  char dirname[128];
  int maxnode,minnode,nstep,iteration;
  float maxtemp,mintemp,*temp,temp_norm;
  float time,timestep,*t_bottom,*t_top;
  int *iop;
  float *x,*y,*z,*vx,*vy,*vz,*sx,*sy,*sz,*sxy,*syz,*szx,*efp,*rho,*rvol;
  float *hv01,*hv02,*hv03,*hv04,*hv05,*hv06,*hv07,*hv08,*hv09;
  float *hv10,*hv11,*hv12,*hv13,*hv14,*hv15;
  FILE *fp;
  LSDADir *dp = NULL;

  if (lsda_cd(handle,"/traleh/metadata") == -1) return 0;
  printf("Extracting TRALEH data\n");

  sprintf(dirname,"/traleh/d000001");
  lsda_cd(handle,dirname);
  lsda_queryvar(handle,"iop",&typid,&length,&filenum);
  num=length;
/*
  allocate memory to read in 1 state
*/
  iop    = (int   *) malloc(num*sizeof(int));
  x      = (float *) malloc(num*sizeof(float));
  y      = (float *) malloc(num*sizeof(float));
  z      = (float *) malloc(num*sizeof(float));
  vx     = (float *) malloc(num*sizeof(float));
  vy     = (float *) malloc(num*sizeof(float));
  vz     = (float *) malloc(num*sizeof(float));
  sx     = (float *) malloc(num*sizeof(float));
  sy     = (float *) malloc(num*sizeof(float));
  sz     = (float *) malloc(num*sizeof(float));
  sxy    = (float *) malloc(num*sizeof(float));
  syz    = (float *) malloc(num*sizeof(float));
  szx    = (float *) malloc(num*sizeof(float));
  efp    = (float *) malloc(num*sizeof(float));
  rho    = (float *) malloc(num*sizeof(float));
  rvol   = (float *) malloc(num*sizeof(float));
  hv01   = (float *) malloc(num*sizeof(float));
  hv02   = (float *) malloc(num*sizeof(float));
  hv03   = (float *) malloc(num*sizeof(float));
  hv04   = (float *) malloc(num*sizeof(float));
  hv05   = (float *) malloc(num*sizeof(float));
  hv06   = (float *) malloc(num*sizeof(float));
  hv07   = (float *) malloc(num*sizeof(float));
  hv08   = (float *) malloc(num*sizeof(float));
  hv09   = (float *) malloc(num*sizeof(float));
  hv10   = (float *) malloc(num*sizeof(float));
  hv11   = (float *) malloc(num*sizeof(float));
  hv12   = (float *) malloc(num*sizeof(float));
  hv13   = (float *) malloc(num*sizeof(float));
  hv14   = (float *) malloc(num*sizeof(float));
  hv15   = (float *) malloc(num*sizeof(float));
/*
  open file and write header
*/
  sprintf(output_file,"%straleh",output_path);
  fp=fopen(output_file,"w");
  write_message(fp,output_file);
  if (!fp) return 0;

  fprintf(fp,"ALE Tracer particle file\n");
  fprintf(fp,"%5d   30\n",num);
  fprintf(fp,"          elemID\n");
  fprintf(fp,"            x            y            z           vx           vy           vz\n");
  fprintf(fp,"           sx           sy           sz          sxy          syz          szx\n");
  fprintf(fp,"          efp          rho         rvol        hisv1        hisv2        hsiv3\n");
  fprintf(fp,"        hisv4        hisv5        hisv6        hisv7        hisv8        hisv9\n");
  fprintf(fp,"       hisv10       hisv11       hisv12       hisv13       hisv14       hisv15\n");
/*
  output_title(handle,"/traleh/metadata",fp);
  output_legend(handle,fp,1,1);
  Loop through time states and write each one
*/
  for(state=1; (dp = next_dir(handle,"/traleh",dp,dirname)) != NULL; state++) {
    if(lsda_read(handle,LSDA_FLOAT,"time",0,1,&time) != 1) break;
    if(lsda_read(handle,LSDA_INT,"iop",0,num,iop) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"x",0,num,x) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"y",0,num,y) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"z",0,num,z) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vx",0,num,vx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vy",0,num,vy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"vz",0,num,vz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sx",0,num,sx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sy",0,num,sy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sz",0,num,sz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"sxy",0,num,sxy) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"syz",0,num,syz) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"szx",0,num,szx) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"efp",0,num,efp) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rho",0,num,rho) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"rvol",0,num,rvol) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv01",0,num,hv01) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv02",0,num,hv02) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv03",0,num,hv03) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv04",0,num,hv04) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv05",0,num,hv05) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv06",0,num,hv06) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv07",0,num,hv07) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv08",0,num,hv08) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv09",0,num,hv09) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv10",0,num,hv10) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv11",0,num,hv11) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv12",0,num,hv12) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv13",0,num,hv13) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv14",0,num,hv14) != num) break;
    if(lsda_read(handle,LSDA_FLOAT,"hv15",0,num,hv15) != num) break;
    fprintf(fp,"%13.5E\n",time);
    for (i=0;i<num;i++) {
      fprintf(fp,"%16d\n",iop[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",x[i],y[i],z[i],vx[i],vy[i],vz[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",sx[i],sy[i],sz[i],sxy[i],syz[i],szx[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",efp[i],rho[i],rvol[i],hv01[i],hv02[i],hv03[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",hv04[i],hv05[i],hv06[i],hv07[i],hv08[i],hv09[i]);
      fprintf(fp,"%13.5E%13.5E%13.5E%13.5E%13.5E%13.5E\n",hv10[i],hv11[i],hv12[i],hv13[i],hv14[i],hv15[i]);
    }
  }
  fclose(fp);

  free(hv15);
  free(hv14);
  free(hv13);
  free(hv12);
  free(hv11);
  free(hv10);
  free(hv09);
  free(hv08);
  free(hv07);
  free(hv06);
  free(hv05);
  free(hv04);
  free(hv03);
  free(hv02);
  free(hv01);
  free(rvol);
  free(rho);
  free(efp);
  free(szx);
  free(syz);
  free(sxy);
  free(sz);
  free(sy);
  free(sx);
  free(vz);
  free(vy);
  free(vx);
  free(z);
  free(y);
  free(x);
  free(iop);

  printf("      %d states extracted\n",state-1);
  return 0;
}
