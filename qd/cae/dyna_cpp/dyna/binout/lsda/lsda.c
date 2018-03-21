/*
  Copyright (C) 2002
  by Livermore Software Technology Corp. (LSTC)
  All rights reserved

NOTE: things to consider working on/fixing some day.  In the open/openmany
routines, I'm not sure what will happen if the file name passed in has
a %XXX on the end already.  In the many case, should also watch for duplicate
names, or overlapping sets (ie, "file file%002 file" should collapse to just
"file").  Probably shouldn't try to expand "file%002" into a list, as it
would either give "file%002%XXX" or maybe "file%003".  But should 002 imply 003?
All I'm sure about at the moment is that things should work correctly if you
pass in a series of distinct base file names, and you want to open EVERYTHING
*/


/* alpha does not seem to have the int64_t type */
#if defined ALPHA || defined NEC
#define int64_t long long
#endif

#define __BUILD_LSDA__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <ctype.h>
#if !defined _WIN32 && !defined WIN64 && !defined MPPWIN
#include <dirent.h>
#include <unistd.h>
#define DIR_SEP '/'
#else
#include <windows.h>
#include <direct.h>
#define DIR_SEP '\\'
#define _errno win_errno
#define int64_t __int64
#endif
#include "lsda.h"

#ifdef _WIN32
#include <windows.h>
#ifndef MAXPATH
#define MAXPATH 2048
#endif
struct dirent {
  char d_name[MAXPATH];
};

typedef struct {
  WIN32_FIND_DATA wfd;
  HANDLE hfind,hfind0;
  struct dirent de;
  char dn_ptr[MAXPATH], filter[MAXPATH];
} DIR;
#undef MAXPATH
EXTERN DIR *opendir(char *spec,char *filter);
EXTERN struct dirent *readdir(DIR *pdir);
EXTERN void rewinddir(DIR *pdir);
EXTERN void closedir(DIR *pdir);
EXTERN int truncate(char *fname, size_t length);

#endif

#include "lsda_internal.h"

#define Offset LSDA_Offset
#define Length LSDA_Length
#define Command LSDA_Command
#define TypeID LSDA_TypeID
typedef unsigned char octet;

#ifdef VISIBLE
#define STATIC
#else
#define STATIC static
#endif

static char *fullfilename(IFile *daf);
STATIC char *finddirmatch(char *name,DIR *dp);
STATIC int write_initialize(LSDAFile *daf,char *salt);
STATIC int read_initialize(LSDAFile *daf,int keepst);
STATIC int rw_initialize(LSDAFile *daf);
STATIC int closeout_var(LSDAFile *daf);
STATIC int lsda_writesymboltable(LSDAFile *daf);
STATIC int lsda_writesymbol(char *ppath,char *curpath,
                           LSDATable *symbol,LSDAFile *daf);
STATIC LSDATable *lsda_readsymbol(LSDAFile *daf);
STATIC int lsda_readsymboltable(LSDAFile *daf);
STATIC void lsda_createbasictypes(LSDAFile *daf);
STATIC void CreateTypeAlias(LSDAFile *daf, char *alias, char *oldtype);
STATIC Length ReadLength(LSDAFile *daf);
STATIC Offset ReadOffset(LSDAFile *daf);
STATIC Command ReadCommand(LSDAFile *daf);
STATIC TypeID ReadTypeID(LSDAFile *daf);
STATIC int ReadSalt(LSDAFile *daf);
STATIC Length ReadData(unsigned char *data, size_t size, size_t count, LSDAFile *daf);
STATIC int WriteSalt(LSDAFile *daf);
STATIC Length WriteData(octet *data, size_t size, size_t count, LSDAFile *daf, int flush);
STATIC void *ReadTrans(LSDAFile *daf,int FileLength,_CF Convert);
STATIC char *findpath(char *from, char *to);
STATIC int lsda_writecd(int handle,char *path);

extern	_CF GetConversionFunction(IFile *ifile,LSDAType *typein, LSDAType *typeout);

STATIC void PruneSymbols(LSDAFile *daf,LSDATable *symbol);
static size_t SymbolSizes(LSDAFile *daf,LSDATable *symbol);
static int alloc_more_daf(int count);
static int lsda_open2(char *filen,int mode,int handle_in,char *key,char *salt);
static char * link_path(char *path,char *link);

static int num_daf = 0;
static LSDAFile *da_store=NULL;
static int _errno = ERR_NONE;
static int report_level = 0;
static char _scbuf[1024];

static int little_i = 1;
#define little_endian (*(char *)(&little_i))

#ifndef HAVE_AES
/*
 Dummy routines so things will link OK
*/
void get_salt(void *s) {}
void aes_enc_key(char *inkey,      int len, aes_ctx *ctx) {}
void aes_enc_blk(char *inblk, char *outblk, aes_ctx *ctx) {}
#endif



#if defined _WIN32 || defined MPPWIN
/*
  Start off with a few UNIX functions Win32 doesn't have....
*/

//int fsync(int fd) {}

DIR *opendir(char *spec,char *filter)
{
/*
  opendir now not only search for the filter matches, but also search for 
  "*" wild card, which suppose to match every body.(if i cant find the matched filter)
*/
  DIR *pdir;
  char myfilter[MAX_PATH];
  char *ptr;

  pdir = (DIR *)malloc(sizeof(DIR));
  memset(pdir, 0, sizeof(DIR));
  _chdir(spec);
  strcpy(myfilter, filter);
  ptr = strrchr(myfilter, '*');  
  if(!ptr) strcat(myfilter, "*");
  pdir->hfind0 = pdir->hfind = FindFirstFile(myfilter, &pdir->wfd);
  if(pdir->hfind0!=(void*)0xFFFFFFFF){
    strcpy(pdir->filter, myfilter);
    strcpy(pdir->dn_ptr, spec);
  }
  else {
    pdir->hfind0 = pdir->hfind = FindFirstFile("*", &pdir->wfd);
    if(pdir->hfind0!=(void*)0xFFFFFFFF) {
      strcpy(pdir->dn_ptr, spec);
      strcpy(pdir->filter, "*");
    }
    else {
      free(pdir);
      return NULL;
    }
  }
  return pdir;
}

void rewinddir(DIR *pdir)
{
  FindClose(pdir->hfind0);
  pdir->hfind0 = pdir->hfind = FindFirstFile(pdir->filter, &pdir->wfd);
}

struct dirent *readdir(DIR *pdir)
{
    if (pdir->hfind) {
      strcpy(pdir->de.d_name, pdir->wfd.cFileName);
      if (!FindNextFile(pdir->hfind, &pdir->wfd))
        pdir->hfind = NULL;
      return &pdir->de;
    }
    return NULL;
}
void closedir(DIR *pdir)
{
    FindClose(pdir->hfind0);  
    free(pdir);
}

int truncate(char *fname, size_t length)
{
  char *tmpfile;
  char buf[1024];
  FILE *fout, *fin;
  int rdsize=0, cursize = 0;
  int breach =0;

  tmpfile = tempnam(NULL, "LSDA");  
  fout = fopen(tmpfile, "wb");
  fin = fopen(fname, "rb");
  if(fout ==NULL || fin == NULL) return -1;
  while(feof(fin)==0)
  {
    rdsize = 1024;
    if(cursize>(int)length)
    {
      rdsize = cursize - length +1024;
      breach = 1;
    }
    rdsize = fread(buf, sizeof(char), rdsize, fin);
    fwrite(buf, sizeof(char), rdsize, fout);
    if(breach) break;
    cursize+=1024;
  }
  fclose(fout);
  fclose(fin);
  rdsize = 1024;
  fout = fopen(fname,"wb");
  fin = fopen(tmpfile,"rb");
  if(fout == NULL || fin == NULL ) return -1;
  while(feof(fin)==0)
  {
    rdsize = fread(buf, sizeof(char), rdsize, fin);
    fwrite(buf, sizeof(char), rdsize, fout);
  }
  fclose(fin);
  fclose(fout);
  return 0;
}
#endif

static char *link_path(
    char *path,     /* Full path of link in data file */
    char *link)     /* value of the link, ie where it points */
{
  /* Return the full path to the thing the link points to.
     This is just simple string manipulation */
  int lp = strlen(path);
  int ll = strlen(link);
  int i,ncomp,nkeep,comp[256];
  char *ret, *cp, *cp2;

  if(link[0]=='/') {  /* simple, link is absolute */
    ret = strdup(link);
    goto cleanup;
  }
  ret = (char *) malloc(lp+ll+2);
  strcpy(ret,path);
  /* path should include the name of the link itself, so remove
     everything after the last / */
  for(i=lp; i>0 && ret[i] != '/'; i--)
    ret[i]=0;
  /* append link contents */
  strcat(ret,link);
cleanup: /* clean up the resulting string */
  /*
    Replace each / in ret with a NULL, saving
    poitners to each component in the path.
  */
  ncomp=0;
  for(i=0; ret[i]; i++)
    if(ret[i]=='/') {
      comp[ncomp++] = i+1;
      ret[i]=0;
    }
  /* Collapse the list of components, as needed,
     removing instances of '.' and '..' and the like.
  */
  nkeep=0;
  for(i=0; i<ncomp; i++) {
    cp = ret + comp[i];
    if(*cp == 0) continue;  /* empty component: must have been // in path */
    if(strcmp(cp,".")==0) continue;  /* remove '.' */
    if(strcmp(cp,"..")==0) {
      if(nkeep > 0) nkeep--;
      continue;
    }
    comp[nkeep++] = comp[i];
  }
  /* rebuild path in place. */
  cp=ret;
  for(i=0; i<nkeep; i++) {
    cp2=ret+comp[i];
    *cp++ ='/';
    while(*cp2) {
      *cp++ = *cp2++;
    }
  }
  if(cp == ret) *cp++ = '/';
  *cp = 0;
  return ret;
}

static IFile *newIFile()
{
  IFile *ret = (IFile *)malloc(sizeof(IFile));
  memset(ret,0,sizeof(IFile));
  return ret;
}

int *_lsda_errno() { return &_errno; }

int lsda_fopen_aes(char *filen,int filenum, Offset offset,int mode, int want,char *key);

int lsda_reopen(char *filen,int filenum, Offset offset,int mode)
{
  int i = -1;
  return lsda_fopen_aes(filen,filenum,offset,mode,i,NULL);
}
int lsda_reopen_aes(char *filen,int filenum, Offset offset,int mode, char *key)
{
  int i = -1;
  return lsda_fopen_aes(filen,filenum,offset,mode, i,key);
}
int lsda_fopen(char *filen,int filenum, Offset offset,int mode,int want)
{
  return lsda_fopen_aes(filen,filenum,offset,mode,want,NULL);
}
int lsda_fopen_aes(char *filen,int filenum, Offset offset,int mode, int want,char *key)
{
  int i,j;
  char lfilen[MAXPATH];

  _errno = ERR_NONE;  /* reset error */

/*
  If the user specified a particular handle, get it if it is available
  else take next available handle.
*/
  if(want < 0) {
    for(i=0; i<num_daf; i++) {
      if(da_store[i].free) break;
    }
    if(i==num_daf && alloc_more_daf(10) < 0) return -1;
  } else {
    if(want >= num_daf) {
      if(alloc_more_daf(want+10-num_daf) < 0) return -1;
    } else {
      if(da_store[want].free == 0) return -1;
    }
    i = want;
  }
  /*
     don't truncate READ type files when reopening them
  */
  if(mode == LSDA_WRITEREAD) mode = LSDA_READWRITE;
  if(mode == LSDA_WRITEONLY ||
     mode == LSDA_APPEND) {
    j=lsda_truncate_aes(filen,filenum,offset,key);
    if(j != LSDA_SUCCESS) return j;
    /* Reset open mode to APPEND if the file length > 0, else WRITEONLY */
    if(filenum > 0 || offset > 0)
      mode=LSDA_APPEND;
    else
      mode=LSDA_WRITEONLY;
  }
  strcpy(lfilen,filen);
  if(filenum > 0) {
    char ext[8];
    sprintf(ext,"%%%3.3d",filenum);
    strcat(lfilen,ext);
  }
  return lsda_open2(lfilen,mode,i,key,NULL);
}

int lsda_truncate(char *filen,int filenum, Offset offset)
{
  return lsda_truncate_aes(filen,filenum,offset,NULL);
}
int lsda_truncate_aes(char *filen,int filenum, Offset offset,char *key)
{
  DIR *dp;
  int i,j,len;
  LSDAFile *daf, dafd;
  char *name;
  unsigned char header[16];
  char basename[64], *cp;
  unsigned char buf[64], *bp;
  int lastnum;
  char tname[8];
  Command cmd;
  LSDAType *type1,*type2;
  Offset loff;
  _CF Convert;

  _errno = ERR_NONE;  /* reset error */
  daf = &dafd;
  InitLSDAFile(daf);
  lsda_createbasictypes(daf);
  daf->maxsize = DEF_MAX_SIZE;
  daf->num_list = 1;
  daf->ifile = (IFile **) malloc(sizeof(IFile *));
  daf->ifile[0] = daf->ifr = daf->ifw = newIFile();
  daf->encrypted = 0;
#ifdef HAVE_AES
  if(key && *key) {
    unsigned char *cp=key;
    unsigned char lkey[16];
    int i;
    for(i=0; i<16; i++) {
      lkey[i]= *cp++;
      if(*cp == 0) cp=key;
    }
    daf->encrypted = 1;
    aes_enc_key(lkey,16,daf->ctx);
  }
#endif
  len = strlen(filen);
  if(filen[len-1] == DIR_SEP) filen[--len]=0;
  for(j=len-1; j>0; j--)
     if(filen[j] == DIR_SEP) {
       daf->ifw->dirname = (char *) malloc(j+1);
       memcpy(daf->ifw->dirname,filen,j);
       daf->ifw->dirname[j]=(char)0;
       daf->ifw->filename = (char *) malloc(len-j+8);
       strcpy(daf->ifw->filename,filen+j+1);
       break;
     }
  if(j == 0) {
    daf->ifw->dirname = (char *) malloc(2);
    strcpy(daf->ifw->dirname,".");
    daf->ifw->filename = (char *) malloc(len+1+8);
    strcpy(daf->ifw->filename,filen);
  }
  if(filenum==0 && offset == 0)
    filenum= -1;
  if(filenum > 0) {
    char ext[8];
    sprintf(ext,"%%%3.3d",filenum);
    strcat(daf->ifw->filename,ext);
  }
  /*
    Make sure the indicated file exists and we can access it, if the offset > 0.
    If the offset==0, just create it.
  */
  if(filenum >= 0 && offset > 0) {
    daf->fpr = daf->fpw = fopen(fullfilename(daf->ifw),"r+b");
    if(daf->fpw == NULL) {   /* indicated file does not exist */
      _errno = ERR_NOFILE;
      if(report_level > 0) fprintf(stderr,"lsda_truncate: file does not exist\n");
      goto cleanup;
    }
    if(daf->encrypted)ReadSalt(daf);
    ReadData(header,1,8,daf);
/*
  Check to make sure offset given is a valid end of symbol table location
*/
    loff=offset-header[1]-header[2]-header[3];
    if(daf->encrypted) loff -= 16;
    if(fseek(daf->fpw,loff,SEEK_SET) != 0){
     _errno = ERR_FSEEK;
     if(report_level > 0) fprintf(stderr,"lsda_truncate: fseek failed\n");
     goto cleanup;
    }
    /* To properly read the "command" field is a bit of a pain.  First, find the
     * proper conversion function, then feed it to the ReadTrans routine, along
     * with the in file size of the field */
    sprintf(tname,"I*%d",header[3]);
    type1 = daf->FindType(daf,tname);
    sprintf(tname,"I*%d",(int) sizeof(Command));
    type2 = daf->FindType(daf,tname);
    daf->ifw->bigendian = header[5];
    daf->ifw->ConvertCommand = GetConversionFunction(daf->ifw,type1,type2);
/*
   For encrypted files, we have to start our read at the beginning
   of the record, to keep in sync.  Also, we want to save all the
   raw (untranslated) data so we can rewrite this record down below.
*/
    if(daf->encrypted) ReadSalt(daf);
    ReadData(buf,header[1],1,daf);
    bp=buf+header[1];
    ReadData(bp,header[3],1,daf);
    Convert = GetConversionFunction(daf->ifr,type1,type2);
    if(Convert)
      Convert(bp,&cmd,1);
    else
      memcpy(&cmd,bp,header[3]);
    if(_errno != ERR_NONE) {
      _errno = ERR_READ;
      if(report_level > 0)
        fprintf(stderr,"lsda_truncate: fread failed to read 1 byte\n");
      goto cleanup;
    }
    if(cmd != LSDA_ENDSYMBOLTABLE) {
     _errno = ERR_NOENDSYMBOLTABLE;
     if(report_level > 0)
       fprintf(stderr,"lsda_truncate: end of symbol table not found\n");
     goto cleanup;
    }
/*
  Set the "next symbol table" offset to 0
*/
    daf->ifw->stoffset = offset - header[2];
    daf->ifw->ateof = 0;
    if(daf->encrypted) {
      if(fseek(daf->fpw,loff,SEEK_SET) != 0){
       _errno = ERR_FSEEK;
       if(report_level > 0) fprintf(stderr,"lsda_truncate: fseek to %ld failed\n",
                            (long) daf->ifw->stoffset);
       goto cleanup;
      }
      /* Rewrite whole record, but with 0s for the offset part */
      memset(bp+header[3],0,header[2]);
      if(WriteSalt(daf) ||
         WriteData(buf,header[1],1,daf,1) != 1 ||
         WriteData(bp,header[3],1,daf,1) != 1 ||
         WriteData(bp+header[3],header[2],1,daf,1) != 1){
       _errno = ERR_WRITE;
       if(report_level > 0) fprintf(stderr,"lsda_truncate: failed to rewrite record\n");
       goto cleanup;
      }

    } else {
      if(fseek(daf->fpw,daf->ifw->stoffset,SEEK_SET) != 0){
       _errno = ERR_FSEEK;
       if(report_level > 0) fprintf(stderr,"lsda_truncate: fseek to %ld failed\n",
                            (long) daf->ifw->stoffset);
       goto cleanup;
      }
      /* have to write the correct number of words -- fortunately since we
       * are writing 0 there is no endianness problem... */
      memset(buf,0,32);
      if(fwrite(buf,header[2],1,daf->fpw) != 1){
       _errno = ERR_WRITE;
       if(report_level > 0) fprintf(stderr,"lsda_truncate: fwrite failed\n");
       goto cleanup;
      }
    }
/*
  Truncate file.
*/
    fclose(daf->fpw);
#if !defined _WIN32
    truncate(fullfilename(daf->ifw),offset);
#endif
    daf->fpr = daf->fpw = NULL;
  }
/*
  remove all files in this series with numbers > filenum
*/
  lastnum = -1;
  strcpy(basename,daf->ifw->filename);
  for(i=0,j=strlen(basename)-1; j>0; j--)
    if(isdigit(basename[j]))
      i=j;
    else
      break;
  if(i && j && basename[j] == '%') basename[j]=0;
#if defined _WIN32 || defined MPPWIN
  dp = opendir(daf->ifw->dirname,basename);
#else
  dp = opendir(daf->ifw->dirname);
#endif
  if(!dp){
   _errno = ERR_OPENDIR;
   if(report_level > 0)
     fprintf(stderr,"lsda_truncate: error opening directory %s\n",daf->ifw->dirname);
   goto cleanup;
  }
  while(name = finddirmatch(basename,dp)) {
    len=strlen(name);
    for(i=0,j=strlen(name)-1; j>0; j--)
      if(isdigit(name[j]))
        i=j;
      else
        break;
    if(name[j] != '%') continue;
    if(i > 0) {
      j = atoi(name+i);
      if(j > lastnum) lastnum = j;
      if(j > filenum && filenum >= 0) {
        sprintf(_scbuf,"%s%c%s%%%3.3d",daf->ifw->dirname,DIR_SEP,basename,j);
        remove(_scbuf);
      }
    }
  }
  closedir(dp);
  if(daf->fpw) fclose(daf->fpw);
  daf->fpr=daf->fpw=NULL;
  daf->FreeTable(daf,daf->top);
  daf->FreeTypes(daf);
  return LSDA_SUCCESS;

cleanup:
  if(daf->fpw) fclose(daf->fpw);
  daf->fpr=daf->fpw=NULL;
  daf->FreeTable(daf,daf->top);
  daf->FreeTypes(daf);
  return -1;
}
int lsda_open_many_aes(char **filen,int num,char *key);

int lsda_open_many(char **filen,int num)
{
  return lsda_open_many_aes(filen,num,NULL);
}
int lsda_open_many_aes(char **filen,int num,char *key)
{
int i,j,k,len,handle;
LSDAFile *daf;
/*
  Get next available handle
*/
_errno = ERR_NONE;  /* reset error */

for(i=0; i<num_daf; i++) {
  if(da_store[i].free) break;
}
if(i==num_daf && alloc_more_daf(10) < 0) return -1;
handle = i;
daf = da_store + i;
InitLSDAFile(daf);
daf->num_list = num;
daf->maxsize = DEF_MAX_SIZE;
/* If we are worried about the user passing in duplicate file names,
 * could go through the list here and wipe out any that are dups */
daf->ifile = (IFile **) malloc(num*sizeof(IFile *));
for(k=0; k<num; k++) {
  len = strlen(filen[k]);
  daf->ifile[k] = daf->ifr = newIFile();
  if(filen[k][len-1] == DIR_SEP) filen[k][--len]=0;
  for(j=len-1; j>0; j--)
     if(filen[k][j] == DIR_SEP) {
       daf->ifr->dirname = (char *) malloc(j+1);
       memcpy(daf->ifr->dirname,filen[k],j);
       daf->ifr->dirname[j]=(char)0;
       daf->ifr->filename = (char *) malloc(len-j+8);
       strcpy(daf->ifr->filename,filen[k]+j+1);
       break;
     }
  if(j == 0) {
    daf->ifr->dirname = (char *) malloc(2);
    strcpy(daf->ifr->dirname,".");
    daf->ifr->filename = (char *) malloc(len+1+8);
    strcpy(daf->ifr->filename,filen[k]);
  }
}
daf->openmode=LSDA_READONLY;
daf->encrypted = 0;
#ifdef HAVE_AES
if(key && *key) {
  unsigned char *cp=key;
  unsigned char lkey[16];
  int i;
  for(i=0; i<16; i++) {
    lkey[i]= *cp++;
    if(*cp == 0) cp=key;
  }
  daf->encrypted = 1;
  aes_enc_key(lkey,16,daf->ctx);
}
#endif
/*
  Open and initialize the file(s)
*/
if(read_initialize(daf,1) >= 0) return handle;
/*
  Some kind of error occured -- free what we allocated and
  get out.
*/
if(daf->ifile) {
  for(k=0; k<num; k++) {
    if(daf->ifile[k]) {
      if(daf->ifile[k]->dirname) free(daf->ifile[k]->dirname);
      if(daf->ifile[k]->filename) free(daf->ifile[k]->filename);
      free(daf->ifile[k]);
    }
  }
  free(daf->ifile);
  daf->ifile = NULL;
}
daf->free = 1;
return -1;
}
STATIC int lsda_checkforsymboltable(LSDAFile *daf)
{
  Command cmd;
  Offset table_pos, end_pos;
  Length stlen;
/*
  Check to see if this looks like a symbol table....
*/
  table_pos = ReadOffset(daf);
  if(table_pos == 0) return 0;   /* end of file -- no more symbol tables */
  if(_errno != ERR_NONE) return -1;
  if(fseek(daf->fpr,table_pos,SEEK_SET) < 0) return -1;
  if(daf->encrypted && ReadSalt(daf)) return -1;
  stlen = ReadLength(daf);
  if(_errno != ERR_NONE) return -1;
  cmd = ReadCommand(daf);
  if(_errno != ERR_NONE) return -1;
  if(cmd != LSDA_BEGINSYMBOLTABLE) return -1;
  end_pos = stlen+table_pos -
            (daf->ifr->FileLengthSize +
             daf->ifr->FileCommandSize +
             daf->ifr->FileOffsetSize);
/*
  The above is OK for encrypted or not -- in encrypted case, "table_pos"
  points to the SALT at the start of the table, which is not included
  in the table length.  But that is OK, because we want to back up over
  the salt when we read the ENDSYMBOLTABLE record...
  if(daf->encrypted) end_pos -= 16;
*/
  if(fseek(daf->fpr,end_pos,SEEK_SET) < 0) return -1;
  /* see if end of symbol table is where it should be */
  if(daf->encrypted && ReadSalt(daf)) return -1;
  stlen = ReadLength(daf);
  if(_errno != ERR_NONE) return -1;
  cmd = ReadCommand(daf);
  if(_errno != ERR_NONE) return -1;
  if(cmd != LSDA_ENDSYMBOLTABLE) return -1;
  /* Leave offset word for next call */
  return 1;
}
int lsda_test_aes(char *filen,char *key);
int lsda_test(char *filen)
{
  return lsda_test_aes(filen,NULL);
}
int lsda_test_aes(char *filen,char *key)
{
  /* check to see if this might be a legit LSDA file */
  unsigned char header[8];
  char tname[8];
  Command cmd;
  LSDAType *type1,*type2;
  int first = 1,len;
  int i,j,retval=0;
  LSDAFile daf0, *daf;

  daf = &daf0;
  _errno = ERR_NONE;
  InitLSDAFile(daf);
  len = strlen(filen);
  daf->ifr = newIFile();
  if(filen[len-1] == DIR_SEP) filen[--len]=0;
  daf->ifr->dirname=(char *)malloc(len+10);
  daf->ifr->filename=(char *)malloc(len+10);
  for(j=len-1; j>0; j--)
    if(filen[j] == DIR_SEP) {
      strcpy(daf->ifr->dirname,filen);
      daf->ifr->dirname[j]=0;
      strcpy(daf->ifr->filename,filen+j+1);
      break;
    }
  if(j == 0) {
    strcpy(daf->ifr->dirname,".");
    strcpy(daf->ifr->filename,filen);
  }
  daf->openmode=LSDA_READONLY;
  daf->encrypted = 0;
#ifdef HAVE_AES
  if(key && *key) {
    unsigned char *cp=key;
    unsigned char lkey[16];
    int i;
    for(i=0; i<16; i++) {
      if(*cp == 0) cp=key;
      lkey[i]= *cp++;
    }
    daf->encrypted = 1;
    aes_enc_key(lkey,16,daf->ctx);
  }
#endif

/*
  Try opening and reading this file.
*/
  if((daf->fpr = fopen(filen,"rb")) == NULL) return 0;  /* failed */
  if(daf->encrypted)ReadSalt(daf);
  if(ReadData(header,1,8,daf) < 8) { /* fail */
    fclose(daf->fpr);
    goto done0;
  }
  daf->ifr->FileLengthSize = header[1];
  daf->ifr->FileOffsetSize = header[2];
  daf->ifr->FileCommandSize= header[3];
  daf->ifr->FileTypeIDSize = header[4];
  daf->ifr->bigendian      = header[5];
  daf->ifr->fp_format      = header[6];
  if(daf->ifr->FileLengthSize  < 1 || daf->ifr->FileLengthSize  > 8 ||
     daf->ifr->FileOffsetSize  < 1 || daf->ifr->FileOffsetSize  > 8 ||
     daf->ifr->FileCommandSize < 1 || daf->ifr->FileCommandSize > 8 ||
     daf->ifr->FileTypeIDSize  < 1 || daf->ifr->FileTypeIDSize  > 8 ||
     daf->ifr->bigendian < 0 || daf->ifr->bigendian > 1) {
    fclose(daf->fpr);
    goto done0;
  }
  lsda_createbasictypes(daf);
/*
  Set conversion functions for length, offset, etc
*/
  sprintf(tname,"I*%d",daf->ifr->FileLengthSize);
  type1 = daf->FindType(daf,tname);
  sprintf(tname,"I*%d",(int) sizeof(Length));
  type2 = daf->FindType(daf,tname);
  daf->ifr->ConvertLength = GetConversionFunction(daf->ifr,type1,type2);

  sprintf(tname,"I*%d",daf->ifr->FileOffsetSize);
  type1 = daf->FindType(daf,tname);
  sprintf(tname,"I*%d",(int) sizeof(Offset));
  type2 = daf->FindType(daf,tname);
  daf->ifr->ConvertOffset = GetConversionFunction(daf->ifr,type1,type2);

  sprintf(tname,"I*%d",daf->ifr->FileCommandSize);
  type1 = daf->FindType(daf,tname);
  sprintf(tname,"I*%d",(int) sizeof(Command));
  type2 = daf->FindType(daf,tname);
  daf->ifr->ConvertCommand = GetConversionFunction(daf->ifr,type1,type2);

  sprintf(tname,"I*%d",daf->ifr->FileTypeIDSize);
  type1 = daf->FindType(daf,tname);
  sprintf(tname,"I*%d",(int) sizeof(TypeID));
  type2 = daf->FindType(daf,tname);
  daf->ifr->ConvertTypeID = GetConversionFunction(daf->ifr,type1,type2);
/*
  OK, now check for what look like reasonable symbol table(s).
*/
  if(daf->encrypted) {
    fseek(daf->fpr,16+header[0],SEEK_SET);  /* skip initial salt */
    ReadSalt(daf);
  } else {
    fseek(daf->fpr,header[0],SEEK_SET);
  }
  ReadLength(daf);
  cmd = ReadCommand(daf);
  if(_errno == ERR_READ ||
    (cmd != LSDA_SYMBOLTABLEOFFSET && cmd != LSDA_ENDSYMBOLTABLE)) {  /* error */
    _errno = ERR_NONE;  /* reset read error */
    goto done1;
  }
  daf->ifr->stoffset=ftell(daf->fpr);
  for(i=0; i<5; i++) {
    j = lsda_checkforsymboltable(daf);
    if(j == -1) break;    /* bad file */
    if(j == 0) i=5;       /* end of file  -- count as OK */
  }
  retval = i < 5 ? 0 : 1;
done1:
  fclose(daf->fpr);
  daf->fpr=NULL;
  daf->FreeTypes(daf);
done0:
  free(daf->ifr->dirname);
  free(daf->ifr->filename);
  free(daf->ifr);
  return retval;
}
int lsda_open(char *filen,int mode)
{
  return lsda_open2(filen,mode,-1,NULL,NULL);
}
int lsda_open_aes(char *filen,int mode,char *key)
{
  return lsda_open2(filen,mode,-1,key,NULL);
}
int lsda_open_salt(char *filen,int mode,char *key,char *salt)
{
  return lsda_open2(filen,mode,-1,key,salt);
}
static int lsda_open2(char *filen,int mode,int handle_in,char *key,char *salt)
{
  int i,j,len,handle;
  LSDAFile *daf;
  char *cp;
  DIR *dp;

  _errno = ERR_NONE;  /* reset error */
/*
  Get next available handle
*/
  if(handle_in < 0) {
    for(i=0; i<num_daf; i++) {
      if(da_store[i].free) break;
    }
    if(i==num_daf && alloc_more_daf(10) < 0) {
      _errno = ERR_MALLOC;
      fprintf(stderr, "lsda_open: memory allocation error");
      return -1;
    }
    handle = i;
  } else {
    i = handle = handle_in;  /* guaranteed to be ok.... */
  }
  daf = da_store + i;
  InitLSDAFile(daf);
  daf->num_list = 1;
  daf->maxsize = DEF_MAX_SIZE;
  daf->ifile = (IFile **) malloc(sizeof(IFile *));
  daf->ifile[0] = newIFile();
  daf->ifr = daf->ifw = daf->ifile[0];

  len = strlen(filen);
  if(filen[len-1] == DIR_SEP) filen[--len]=0;
  for(j=len-1; j>0; j--)
     if(filen[j] == DIR_SEP) {
       daf->ifr->dirname = (char *) malloc(j+1);
       memcpy(daf->ifr->dirname,filen,j);
       daf->ifr->dirname[j]=(char)0;
       daf->ifr->filename = (char *) malloc(len-j+8);
       strcpy(daf->ifr->filename,filen+j+1);
       break;
     }
  if(j == 0) {
    daf->ifr->dirname = (char *) malloc(2);
    strcpy(daf->ifr->dirname,".");
    daf->ifr->filename = (char *) malloc(len+1+8);
    strcpy(daf->ifr->filename,filen);
  }
  daf->openmode=mode;
  daf->encrypted = 0;
#ifdef HAVE_AES
  if(key && *key) {
    unsigned char *cp=key;
    unsigned char lkey[16];
    int i;
    for(i=0; i<16; i++) {
      if(*cp == 0) cp=key;
      lkey[i]= *cp++;
    }
    daf->encrypted = 1;
    aes_enc_key(lkey,16,daf->ctx);
  }
#endif
  /*
    Open and initialize the file
  */
  switch(mode) {
     case LSDA_READONLY:  /* open existing file and preserve data */
       if(read_initialize(daf,1) < 0) goto cleanup;
       return handle;
     case LSDA_READWRITE:  /* open existing file and preserve data */
       if(read_initialize(daf,1) < 0) goto cleanup;
       rw_initialize(daf);
       return handle;
     case LSDA_APPEND:     /* open existing file for writing */
       if(read_initialize(daf,0) > 0) {
         rw_initialize(daf);
         daf->openmode=LSDA_WRITEONLY;
         return handle;
       }
       /*
         file not found, so create WRITEONLY by
         falling through to next case.
         Reset openmode and the file handles (which were set to NULL
         by read_initialize)
       */
       daf->openmode=mode=LSDA_WRITEONLY;
       daf->ifr = daf->ifw = daf->ifile[0];
     case LSDA_WRITEONLY:  /* create file */
     case LSDA_WRITEREAD:  /* create file */
#if defined _WIN32 || defined MPPWIN
       if((dp = opendir(daf->ifw->dirname,daf->ifw->filename)) == NULL) {
#else
       if((dp = opendir(daf->ifw->dirname)) == NULL) {
#endif
         _errno = ERR_OPENDIR;
         if(report_level > 0) fprintf(stderr,
      "lsda_open: Cannot open directory %s\nCheck permissions\n",daf->ifr->dirname);
         goto cleanup;
       }
       while(cp=finddirmatch(daf->ifw->filename,dp)) {
         remove(cp);
       }
       closedir(dp);
       if((daf->fpw = fopen(filen,"w+b"))==NULL) {
         _errno = ERR_OPENFILE;
         if(report_level > 0) fprintf(stderr,
      "lsda_open: Cannot open file %s\nCheck permissions\n",filen);
         goto cleanup;
       }
       if(write_initialize(daf,salt)<0) goto cleanup;
       return handle;
    }
cleanup:
  if(daf->ifile) {
    if(daf->ifr) {
      if(daf->ifr->dirname) free(daf->ifr->dirname);
      if(daf->ifr->filename) free(daf->ifr->filename);
      free(daf->ifr);
    }
    free(daf->ifile);
    daf->ifile = NULL;
  }
  daf->free = 1;
  return -1;
}

static int alloc_more_daf(int count)
{
  int i;
  if(da_store)
    da_store = (LSDAFile *) realloc(da_store,(num_daf+count)*sizeof(LSDAFile));
  else
    da_store = (LSDAFile *) malloc(count*sizeof(LSDAFile));
  if(!da_store){
    _errno = ERR_MALLOC;
    if(report_level > 0)
      fprintf(stderr,"alloc_more_daf: malloc of %d failed\n",count);
    return -1;
  }
  for(i=num_daf ; i<num_daf+count; i++)
    da_store[i].free = 1;
  num_daf += count;
  return 1;
}

STATIC int write_initialize(LSDAFile *daf,char *salt)
{
unsigned char header[16];
int handle = daf-da_store;
Length rlen;
Command cmd;
Offset offset;

header[0] = 8;               /* number of bytes in header, this included */
header[1] = sizeof(Length);  /* size of int used in data record lengths */
header[2] = sizeof(Offset);  /* size of int used in data file offsets */
header[3] = sizeof(Command); /* size of int used in data file commands */
header[4] = sizeof(TypeID);  /* size of int used in data file typeids */
header[5] = little_endian;   /* 0 for bigendian, 1 for little_endian */
header[6] = FP_FORMAT;       /* 0 = IEEE */
header[7] = 0;

fseek(daf->fpw,0,SEEK_SET);

if(daf->encrypted) {
  if(salt) {  /* use user supplied salt and skip the encryption step */
    memcpy(daf->salt,salt,16);
    if(fwrite(daf->salt, 1, 16, daf->fpw) != 16) goto write_error;
  } else {
    get_salt(daf->salt);  /* first time, need to initialize salt */
    if(WriteSalt(daf)) goto write_error;
  }
}
if(WriteData(header,1,8,daf,1) < 8) goto write_error;
/*
  Create empty space for symbol table pointer
*/
rlen = sizeof(Length)+sizeof(Command)+sizeof(Offset);
cmd = LSDA_SYMBOLTABLEOFFSET;
if(daf->encrypted && WriteSalt(daf)) goto write_error;
if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto write_error;
if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto write_error;
offset = 0;
daf->ifw->stoffset = ftell(daf->fpw);
daf->ifw->ateof = 0;
if(WriteData((octet *) &offset,sizeof(Offset),1,daf,1) < 1) goto write_error;
if(lsda_cd(handle,"/") < 0) { fclose(daf->fpw); daf->fpw=NULL; daf->ifw=NULL; return -1; }
if(lsda_writecd(handle,"/") < 0) { fclose(daf->fpw); daf->fpw=NULL; daf->ifw=NULL; return -1; }
strcpy(daf->lastpath,"/");
daf->ifw->FileLengthSize = header[1];
daf->ifw->FileOffsetSize = header[2];
daf->ifw->FileCommandSize= header[3];
daf->ifw->FileTypeIDSize = header[4];
daf->ifw->bigendian      = header[5];
daf->ifw->fp_format      = header[6];
lsda_createbasictypes(daf);
return 1;

write_error:
 _errno = ERR_WRITE;
 if(report_level > 0) {
   fprintf(stderr,
  "write_initialize: Write error on file %s\n",fullfilename(daf->ifw));
 }
 if(daf->fpw) fclose(daf->fpw);
 daf->fpw=NULL;
 daf->ifw=NULL;
 return -1;
}

STATIC void lsda_createbasictypes(LSDAFile *daf)
{
LSDAType *type;
char tname[32];
/*
  Create the necessary intrinsic types
*/
 if(daf->ntypes > 0) return;  /* have already been in here... */
 type = daf->CreateType(daf,"I*1");
  type->length_on_disk = type->length = 1;
 type = daf->CreateType(daf,"I*2");
  type->length_on_disk = type->length = 2;
 type = daf->CreateType(daf,"I*4");
  type->length_on_disk = type->length = 4;
 type = daf->CreateType(daf,"I*8");
  type->length_on_disk = type->length = 8;
 type = daf->CreateType(daf,"U*1");
  type->length_on_disk = type->length = 1;
 type = daf->CreateType(daf,"U*2");
  type->length_on_disk = type->length = 2;
 type = daf->CreateType(daf,"U*4");
  type->length_on_disk = type->length = 4;
 type = daf->CreateType(daf,"U*8");
  type->length_on_disk = type->length = 8;
 type = daf->CreateType(daf,"R*4");
  type->length_on_disk = type->length = 4;
 type = daf->CreateType(daf,"R*8");
  type->length_on_disk = type->length = 8;
 type = daf->CreateType(daf,"LINK");
  type->length_on_disk = type->length = 1;
/*
  And type aliases
*/
  sprintf(tname,"I*%d",(int) sizeof(int));
  CreateTypeAlias(daf,"int",tname);

  sprintf(tname,"I*%d",(int) sizeof(short));
  CreateTypeAlias(daf,"short",tname);

  sprintf(tname,"I*%d",(int) sizeof(long));
  CreateTypeAlias(daf,"long",tname);

  sprintf(tname,"U*%d",(int) sizeof(unsigned int));
  CreateTypeAlias(daf,"uint",tname);

  sprintf(tname,"U*%d",(int) sizeof(unsigned short));
  CreateTypeAlias(daf,"ushort",tname);

  sprintf(tname,"U*%d",(int) sizeof(unsigned long));
  CreateTypeAlias(daf,"ulong",tname);

  sprintf(tname,"R*%d",(int) sizeof(float));
  CreateTypeAlias(daf,"float",tname);

  sprintf(tname,"R*%d",(int) sizeof(double));
  CreateTypeAlias(daf,"double",tname);

  sprintf(tname,"I*%d",(int) sizeof(FortranInteger));
  CreateTypeAlias(daf,"integer",tname);

  sprintf(tname,"R*%d",(int) sizeof(FortranReal));
  CreateTypeAlias(daf,"real",tname);

  sprintf(tname,"R*%d",(int) sizeof(FortranDouble));
  CreateTypeAlias(daf,"double precision",tname);
}
STATIC void CreateTypeAlias(LSDAFile *daf, char *alias, char *oldtype)
{
LSDAType *otype, *ntype;

  ntype = daf->CreateType(daf,alias);
  otype = daf->FindType(daf,oldtype);
  ntype->alias = otype;
}
STATIC int read_initialize(LSDAFile *daf,int keepst)
{
/*
  Read in the existing symbol table (or reconstruct it as needed....someday)
  If keepst==0, don't actually keep ST data.  But we DO read it to make
  sure the file is valid.
*/
unsigned char header[8];
char tname[8];
Command cmd;
LSDAType *type1,*type2;
char *name,fullname[1024];
char base_directory[2048];
IFile *ifile;
int namelen;
int retval= -1;
int i,j,is_newfile;
DIR *dp;
int org_num_list = daf->num_list;

// Extract directory name - needed for a hands on bugfix
int last_separator = -1;
for(int ii=0; ii<strlen(daf->ifile[0]->filename); ++ii){
    if((daf->ifile[0]->filename[ii] == '/') || (daf->ifile[0]->filename[ii] == '\\')){
        last_separator = ii;
    }
}
if(last_separator >= 0){
    strncpy(base_directory,daf->ifile[0]->filename,last_separator);
    base_directory[last_separator] = '\0';
} else {
    sprintf(base_directory,".");
}


for(i=0; i==0 || i<org_num_list; i++) {

  daf->ifr = daf->ifile[i];
  namelen = strlen(daf->ifile[i]->filename);

#if defined _WIN32 || defined MPPWIN
  dp = opendir(daf->ifile[i]->dirname,daf->ifile[i]->filename);
#else
  dp = opendir(daf->ifile[i]->dirname);
#endif

  if(dp == NULL) {
    _errno = ERR_OPENDIR;
    if(report_level > 0) fprintf(stderr,
    "read_initialize: Cannot open directory %s\nCheck permissions\n",
    daf->ifile[i]->dirname);
    return -1;
  }
/*
  Try opening and reading all files of the form filename[%digits]
  where the [%digits] are optional.  As far as I can see, I
  don't really care what order we open them in.  But remember which one
  is the highest numbered -- that is the one we will do any writing to.
*/
  while(daf->ifile[i] && (name = finddirmatch(daf->ifile[i]->filename,dp))) {
    if(strlen(name) == namelen) {  /* opened base file */
      ifile = daf->ifile[i];
      is_newfile = 0;
    } else {
      ifile = newIFile();
      ifile->dirname = (char *) malloc(strlen(daf->ifile[i]->dirname)+1);
      ifile->filename = (char *) malloc(strlen(name)+1);
      strcpy(ifile->dirname,daf->ifile[i]->dirname);
      strcpy(ifile->filename,name);
      is_newfile = 1;
    }
    //sprintf(fullname,"%s%c%s",ifile->dirname,DIR_SEP,ifile->filename); // buggy
    sprintf(fullname,"%s%c%s",base_directory,DIR_SEP,ifile->filename);
    //sprintf(fullname,"G:/Programming/CPP/Lasso-CAE-Analytics/src/io/lsda/binout");
    if((daf->fpr = fopen(fullname,"rb")) == NULL) { /* skip this file */
      free(ifile->dirname);
      free(ifile->filename);
      free(ifile);
      if(!is_newfile) daf->ifile[i] = NULL;
      continue;
    }
    if(daf->encrypted) ReadSalt(daf);
    if(ReadData(header,1,8,daf) < 8) { /* skip this file */
      fclose(daf->fpr);
      free(ifile->dirname);
      free(ifile->filename);
      free(ifile);
      if(!is_newfile) daf->ifile[i] = NULL;
      daf->fpr=NULL;
      continue;
    }

    ifile->FileLengthSize = header[1];
    ifile->FileOffsetSize = header[2];
    ifile->FileCommandSize= header[3];
    ifile->FileTypeIDSize = header[4];
    ifile->bigendian      = header[5];
    ifile->fp_format      = header[6];
    lsda_createbasictypes(daf);
/*
  Set conversion functions for length, offset, etc
*/
    sprintf(tname,"I*%d",ifile->FileLengthSize);
    type1 = daf->FindType(daf,tname);
    sprintf(tname,"I*%d",(int) sizeof(Length));
    type2 = daf->FindType(daf,tname);
    ifile->ConvertLength = GetConversionFunction(ifile,type1,type2);

    sprintf(tname,"I*%d",ifile->FileOffsetSize);
    type1 = daf->FindType(daf,tname);
    sprintf(tname,"I*%d",(int) sizeof(Offset));
    type2 = daf->FindType(daf,tname);
    ifile->ConvertOffset = GetConversionFunction(ifile,type1,type2);

    sprintf(tname,"I*%d",ifile->FileCommandSize);
    type1 = daf->FindType(daf,tname);
    sprintf(tname,"I*%d",(int) sizeof(Command));
    type2 = daf->FindType(daf,tname);
    ifile->ConvertCommand = GetConversionFunction(ifile,type1,type2);

    sprintf(tname,"I*%d",ifile->FileTypeIDSize);
    type1 = daf->FindType(daf,tname);
    sprintf(tname,"I*%d",(int) sizeof(TypeID));
    type2 = daf->FindType(daf,tname);
    ifile->ConvertTypeID = GetConversionFunction(ifile,type1,type2);

/*
  Read in symbol table
  Should put reconstruction code in here eventually...
*/
    if(daf->encrypted) {
      fseek(daf->fpr,16+header[0],SEEK_SET);  /* skip initial salt */
      ReadSalt(daf);
    } else {
      fseek(daf->fpr,header[0],SEEK_SET);
    }
    daf->ifr = ifile;  /* so ReadTrans routines will work */
    ReadLength(daf);
    cmd = ReadCommand(daf);
    if(_errno == ERR_READ || 
    (cmd != LSDA_SYMBOLTABLEOFFSET && cmd != LSDA_ENDSYMBOLTABLE)) {  /* skip this file for now */
      _errno = ERR_NONE;  /* reset read error */
      if(report_level > 0) {
        fprintf(stderr,"Error reading symbol table in file %s\n",name);
        fprintf(stderr,"  Skipping this file\n");
      }
      free(ifile->dirname);
      free(ifile->filename);
      free(ifile);
      if(!is_newfile) daf->ifile[i] = NULL;
      fclose(daf->fpr);
      daf->fpr=NULL;
      continue;
    }
    daf->ifr->stoffset=ftell(daf->fpr);
    if(lsda_readsymboltable(daf) == 1) { /* OK, keep this one */
      if(is_newfile) {
        daf->ifile = (IFile **) realloc(daf->ifile,(daf->num_list+1)*sizeof(IFile *));
        daf->ifile[daf->num_list++] = ifile;
      }
      retval=1;
      if(!keepst) PruneSymbols(daf,daf->top);
    } else {
      free(ifile->dirname);
      free(ifile->filename);
      free(ifile);
      if(!is_newfile) daf->ifile[i] = NULL;
    }
    fclose(daf->fpr);
    daf->fpr=NULL;
  }
  closedir(dp);
}
daf->ifw = NULL;
daf->fpw = NULL;
daf->ifr = NULL;
daf->fpr = NULL;
daf->stpendlen = 0;
daf->cwd = daf->top;
/*
 * In case we had problems opening one or more of the files, reduce the ifile
 * list here
 */
for(i=j=0; i<daf->num_list; i++) {
  if(daf->ifile[i] != NULL)
    daf->ifile[j++] = daf->ifile[i];
}
daf->num_list = j;
if(retval < 0) _errno=ERR_OPENFILE;
return retval;
}
STATIC int rw_initialize(LSDAFile *daf)
{
/*
  Routine to do the "write" part of the initialization of a READWRITE
  open call.  read_initialize has already been called.

  steps: put highest numbered file last in ifile list.  Check to see
  if it is compatible.  If so, use it, if not, open another one.
*/
IFile *ifile;
int i,largest,index,val;
char *cp;

/*
  Find the file with the largest extension number
*/
  largest = -1;
  index = -1;
  for(i=0; i<daf->num_list; i++) {
    cp = strrchr(daf->ifile[i]->filename,'%');  /* find % */
    if(!cp) {
       val=0;
    } else    {
      val = atoi(cp+1);                /* convert following to number */
      for(cp++; *cp; cp++)
        if(!isdigit(*cp)) val=0;       /* but only if all following are digits */
    }
    if(val > largest) {
      largest=val;
      index = i;
    }
  }
/*
  If it is not last in the list, make it last in the list
*/
  if(index < daf->num_list - 1) {
    ifile = daf->ifile[index];
    daf->ifile[index] = daf->ifile[daf->num_list-1];
    daf->ifile[daf->num_list-1] = ifile;
  } else {
    ifile = daf->ifile[daf->num_list-1];
  }
/*
  Check to see if this file is compatible with the way I want to write
  it
*/
  daf->ifw = ifile;
  daf->ifw->ateof = 0;
  daf->npend = 0;
  daf->continued = 0;
  daf->stpendlen = 0;
  if(ifile->FileLengthSize == sizeof(Length) &&
     ifile->FileOffsetSize == sizeof(Offset) &&
     ifile->FileCommandSize== sizeof(Command) &&
     ifile->FileTypeIDSize == sizeof(TypeID) &&
     ifile->bigendian      == little_endian &&
     ifile->fp_format      == FP_FORMAT) {  /* file is compatible, use it */
    daf->fpw = fopen(fullfilename(ifile),"r+b");
    if(daf->fpw == NULL) {
      daf->ifw = NULL;
      return -1;
    }
    return 0;
  }
  return lsda_nextfile(daf-da_store);
}

STATIC int closeout_var(LSDAFile *daf)
{
  Length len = ftell(daf->fpw) - daf->var->offset;
  Command cmd;
  TypeID tid;
  char nlen;
/*
  If encryption, have to rewrite the whole header, not just the
  length
*/
  daf->ifw->ateof = 0;

  if(fseek(daf->fpw,daf->var->offset,SEEK_SET) < 0) {
    _errno = ERR_FSEEK;
    if(report_level > 0) {
      fprintf(stderr,"closeout_var: seek error on file %s\n",
              fullfilename(daf->ifw));
    }
    return -1;
  }
  if(daf->encrypted) {
    cmd=LSDA_DATA;
    nlen = strlen(daf->var->name);
    len -= 16;  /* don't include length of first salt in record length */
    tid = LSDAId(daf->var->type);
    if(WriteSalt(daf) ||     /* write new salt value */
       WriteData((octet *) &len,sizeof(Length),1,daf,1) != 1 ||
       WriteData((octet *) &cmd,sizeof(Command),1,daf,1) != 1 ||
       WriteData((octet *) &tid,sizeof(TypeID),1,daf,1) != 1 ||
       WriteData((octet *) &nlen,1,1,daf,1) != 1 ||
       WriteData((octet *) daf->var->name,nlen,1,daf,1) != 1) {
      _errno = ERR_WRITE;
      if(report_level > 0) {
        fprintf(stderr,"closeout_var: write error on file %s\n",
                fullfilename(daf->ifw));
      }
      return -1;
    }
    len -= 16;  /* don't include length of second salt in variable length */
  } else {
    if(WriteData((octet *) &len,sizeof(Length),1,daf,1) < 1) {
      _errno = ERR_WRITE;
      if(report_level > 0) {
        fprintf(stderr,"closeout_var: write error on file %s\n",
                fullfilename(daf->ifw));
      }
      return -1;
    }
  }
  daf->continued = 0;
  daf->var->length = (len-sizeof(Length)-sizeof(Command)-sizeof(TypeID)-
     strlen(daf->var->name)-1)/LSDASizeOf(daf->var->type);
  return 1;
}

STATIC int SwitchFamilyMember(LSDAFile *daf,LSDATable *var)
{
  /* This is only ever called while reading */
  if(daf->fpr && (daf->fpw != daf->fpr)) fclose(daf->fpr);
  daf->ifr = var->ifile;
  if(daf->ifr == daf->ifw) {
    daf->fpr = daf->fpw;
  } else {
/*
  Opening file read/write: if the user calls lsda_rewrite, we could
  need to write to a pre-existing file.
*/
    if((daf->fpr = fopen(fullfilename(daf->ifr),"r+b")) == NULL) {
      _errno = ERR_OPENFILE;
      if(report_level > 0)
        fprintf(stderr,"lsda_SwitchFamilyMember: error opening %s",fullfilename(daf->ifr));
      return -1;
    }
  }
  return 1;
}

Length lsda_write(int handle, int type_id,char *name,Length length,void *data)
{
  LSDAFile *daf;
  int tsize;
  TypeID tid;
  Length rlen;
  Command cmd = LSDA_DATA;
  char nlen;
  LSDATable *var,*pvar;
  LSDAType *type;
  char prevpath[MAXPATH],cwd[MAXPATH];
  int j,retval;
  char lname[256],ldir[256];

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_write: invalid handle %d",handle);
    return -1;
  }
  daf = da_store + handle;

  if(name[0]==0) {  /* continue writing previous variable */
    if(!daf->var)  {
      _errno = ERR_NOCONT;
      if(report_level > 0)
        fprintf(stderr,"Empty variable name used while not currently writing a variable\n");
      return -1;
    }
    daf->continued = 1;
    tsize = LSDASizeOf(daf->var->type);
    retval = WriteData((octet *) data,tsize,length,daf,0);
    if(retval < length) _errno = ERR_WRITE;
    daf->var->length += retval;
    return retval;
  }
  cwd[0]=0;
/*
  Writing new variable.  If were not finished with the old one,
  close it out.
*/
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) return -1;
  }
  if(!daf->ifw->ateof) {
    fseek(daf->fpw,0,SEEK_END);
    daf->ifw->ateof = 1;
  }
/*
  If we are getting too big, wrap to a new file
*/
  if(ftell(daf->fpw)+daf->stpendlen > daf->maxsize)
    lsda_nextfile(handle);
/*
  Check for directory portion in variable name
*/
  nlen = strlen(name);
  for(j=nlen-1; j>0; j--)
     if(name[j] == '/') {
       strcpy(ldir,name);
       ldir[j]=(char)0;
       strcpy(lname,name+j+1);
       break;
     }
  if(j == 0) {
    strcpy(lname,name);
  } else {
    strcpy(cwd,daf->GetCWD(daf));
    lsda_cd(handle,ldir);
  }
/*
  Update CWD in file if necessary
*/
  if(daf->pathchanged) {
    strcpy(prevpath,daf->lastpath);
    strcpy(daf->lastpath,daf->GetCWD(daf));
    if(lsda_writecd(handle,findpath(prevpath,daf->lastpath)) < 0) {
      if(report_level > 0) fprintf(stderr,"lsda_write: updating CWD\n");
      if(cwd[0])lsda_cd(handle,cwd);
      return -1;
    }
  }
  if((type=daf->FindTypeByID(daf,type_id)) == NULL) {
    _errno = ERR_DATATYPE;
    if(report_level > 0) fprintf(stderr,
        "lsda_write: unrecognized data type %d\n",type_id);
    if(cwd[0])lsda_cd(handle,cwd);
    return -1;
  }
  var=daf->CreateVar(daf,type,lname);
  var->offset = (Offset) ftell(daf->fpw);
  var->length = (Length) length;
  var->ifile = daf->ifw;
  /* mark variable and all its parents dirty, so they will get checked
     when it is time to dump this into the symbol table */
  for(pvar=var; pvar; pvar=pvar->parent)
     pvar->dirty=1;
  nlen = (char) strlen(var->name);
  daf->stpendlen += sizeof(Length)+sizeof(Command)+sizeof(TypeID)+nlen+
                    sizeof(Offset)+sizeof(Length);
  daf->var=var;

  tsize = LSDASizeOf(type);
  tid = LSDAId(type);
  rlen = sizeof(Length)+sizeof(Command)+sizeof(TypeID)+nlen+1+length*tsize;
  if(daf->encrypted) {
    rlen += 16;   /* increase record length to account for salt before data */
    if(daf->encrypted && WriteSalt(daf)) goto write_error;
  }
  if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto write_error;
  if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto write_error;
  if(WriteData((octet *) &tid,sizeof(TypeID) ,1,daf,1) < 1) goto write_error;
/*
  The variable name is stored as a 1 char length and then a non-terminated
  string
*/
  if(WriteData((octet *) &nlen,1,1,daf,1) < 1) goto write_error;
  if(WriteData((octet *) lname,1,(int)nlen,daf,1) < nlen) goto write_error;
  if(daf->encrypted && WriteSalt(daf)) goto write_error;
  retval = WriteData((octet *) data,tsize,length,daf,0);
  if(retval < length) _errno = ERR_WRITE;
  if(cwd[0])lsda_cd(handle,cwd);
  return retval;

write_error:
   _errno = ERR_WRITE;
   if(report_level > 0) {
     fprintf(stderr,
        "lsda_write: write error on file %s\n",fullfilename(daf->ifw));
   }
   if(cwd[0])lsda_cd(handle,cwd);
   return -1;
}

Length lsda_rewrite(int handle, int type_id, char *name,Length offset, Length number,void *data)
/*
  Rewrite part of an existing variable.  For now type_id is ignored -- it is assumed the
  user knows what they are doing and is writing exactly the same type of data
  as was in the file originally.  Someday we should allow a different type
  input, and convert to the proper output type on the fly.
*/
{
  LSDAFile *daf;
  LSDATable *var;
  Offset foffset;
  int j,retval;
  int tsizedisk;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_rewrite: invalid handle %d",handle);
    return -1;
  }
  daf = da_store + handle;
  if(daf->openmode != LSDA_READWRITE && daf->openmode != LSDA_WRITEREAD) {
    _errno = ERR_WRITE;
    if(report_level > 0)
      fprintf(stderr,
        "lsda_rewrite: File %s\n is must be opened READWRITE or WRITEREAD\n",
         daf->ifw ? daf->ifw->filename : NULL);
    return -1;
  }
/*
  Find the existing variable they want to rewrite.
*/
  var=daf->FindVar(daf,name,0,1);
  if(var == NULL) {
    _errno = ERR_NOVAR;
    if(report_level > 0)
      fprintf(stderr,
        "lsda_rewrite: variable %s not found while writing file %s\n CWD=%s\n",
         name,daf->ifw ? daf->ifw->filename : NULL,daf->GetCWD(daf));
    return -1;
  }

  if(offset >= var->length) return 0;
  if(offset+number > var->length) number = var->length - offset;

  tsizedisk = LSDASizeOf(var->type);
  if(daf->encrypted && (16 % tsizedisk)) {
  /*
    This cannot happen at the moment -- no data types are this large...
  */
    fprintf(stderr,"Error: rewriting varaible %s/%s in LSDA file %s:",
    daf->GetCWD(daf),name,daf->ifr->filename);
    fprintf(stderr,"       File is encrypted and variable size does not divide 16\n");
    _errno = ERR_READ;
    return 0;
  }
/*
  Writing a variable.  If were not finished with the last one,
  close it out.
*/
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) return -1;
  }
  if(((daf->fpr == NULL) || (var->ifile != daf->ifr))
        && SwitchFamilyMember(daf,var) < 0) {
    if(report_level > 0) fprintf(stderr,"lsda_rewrite: error\n");
    return -1;
  }
  daf->ifr->ateof = 0;
  if(daf->encrypted) {  /* have encryption */
    char salt[32],salt2[16];
    char buf[16],tmpblk[16];
    int i,enddata;
    octet *dp = (octet *) data;
    octet *dextra = 0;
    int spill1,spill2,gap1,gap2,nwhole;
    Offset voffset,start,finish;
    retval=number;  /* unless we hit an error... */
/*
  Offset to start of actual data for this variable.  Include
  the 16 byte salt before and after the header
*/
    voffset = 16+var->offset +var->ifile->FileLengthSize+
              var->ifile->FileCommandSize+var->ifile->FileTypeIDSize+
              strlen(var->name)+1+16;
/*
  Determine # extra bytes at the beginning and end to get us to
  16 byte boundaries, plus # of whole blocks between
*/
    start=offset*tsizedisk;
    finish=start+number*tsizedisk;
    gap1   = start % 16;         /* bytes between block boundary and start of data */
    spill1 = (16 - gap1) % 16;   /* bytes between start of data and next boundary */
    spill2 = finish % 16;        /* bytes between block boundary and end of data */
    gap2   = (16 - spill2) % 16; /* bytes between end of data and next boundar */
    nwhole = ((finish-spill2)-(start+spill1)) / 16; /* # whole blocks our data spans */
    enddata=(var->length-offset-number)*tsizedisk;  /* # bytes left to deal with at the end */
/*
  If nwhole<1, check to see if we begin and end in a single block
  Simplify by reading in and decrypting this block, modifying
  it, and then pretending that that is what was passed in.

  Also, read and keep the existing file data just before the
  end of what we will be writing, so we can decrypt all following
  data and re-encrypt it afterward.
*/
    if(nwhole == -1 || (nwhole ==0 && (gap1 == 0 || gap2 == 0))) {
      foffset = voffset+start-gap1-16;
      fseek(daf->fpr,foffset,SEEK_SET);
      fread(salt,1,32,daf->fpr);
      aes_enc_blk(salt,buf,daf->ctx);
      for(i=0; i<16; i++) {
        salt2[i] = salt[i+16];    /* save for dealing with dextra below */
        tmpblk[i] = salt[i+16] ^ buf[i];  /* get decrypted data from file */
      }
      for(i=gap1; i<(16-gap2); i++)  /* fill in values user wants to change */
        tmpblk[i] = *dp++;
      start -= gap1;            /* fake data so this tmpblk gets written out */
      gap1 = spill1 = 0;
      finish += gap2;
      if(finish > var->length*tsizedisk) {  /* var ends before next boundary */
        finish = var->length*tsizedisk;
        spill2 = finish % 16;
        gap2   = (16 - spill2) % 16;
        enddata = 0;
        nwhole=0;
      } else {
        enddata -= gap2;
        gap2 = spill2 = 0;
        nwhole=1;
      }
      dp = tmpblk;
    } else if(enddata) {    /* read in salt for decrypting dextra below */
      foffset = voffset+finish-spill2-16;
      fseek(daf->fpr,foffset,SEEK_SET);
      fread(salt2,1,16,daf->fpr);
    }
/*
   Read salt and deal with any extra bytes at the beginning
   to get things going
*/
    if(gap1 == 0) {
      foffset = voffset+start-16;
      fseek(daf->fpr,foffset,SEEK_SET);
      fread(salt,1,16,daf->fpr);
      fseek(daf->fpr,foffset+16,SEEK_SET);  /* switching from read to write */
    } else {
      foffset = voffset+start-16-gap1;
      fseek(daf->fpr,foffset,SEEK_SET);
      fread(salt,1,16+gap1,daf->fpr);
      aes_enc_blk(salt,buf,daf->ctx);
      for(i=0; i<gap1; i++)
        salt[i] = salt[16+i];
      for(; i<16; i++)
        salt[i] = (*dp++) ^ buf[i];
      foffset = voffset+start;
      fseek(daf->fpr,foffset,SEEK_SET);
      fwrite(salt+gap1,1,spill1,daf->fpr);
    }
    for(j=0; j<nwhole; j++) {       /* deal with any whole blocks */
      aes_enc_blk(salt,buf,daf->ctx);
      for(i=0; i<16; i++)
        salt[i] = (*dp++) ^ buf[i];
      fwrite(salt,1,16,daf->fpr);
    }
    if(enddata) {         /* there is data to be re-encrypted */
      int n=(enddata+spill2)/16+1;
/*
   Don't want to process this in a rolling fashion because that would
   require a lot of fseek calls -- much better to just read in all
   the rest of this data.
   Don't worry if the variable does not end on a 16 byte boundary --
   and garbage at the end (including read failure due to EOF) will
   end up not actually being used.
*/
      dextra = (octet *) malloc(16*n);
      foffset = voffset+finish-spill2;
      fseek(daf->fpr,foffset,SEEK_SET);
      fread(dextra,1,16*n,daf->fpr);
      for(j=0; j<n; j++) {                /* read data and decrypt it */
        aes_enc_blk(salt2,buf,daf->ctx);
        for(i=0; i<16; i++) {
          salt2[i] = dextra[16*j+i];
          dextra[16*j+i] = salt2[i] ^ buf[i];
        }
      }
      for(j=0; j<spill2; j++)   /* overwrite head of decrypted buffer with new data */
        dextra[j] = *dp++;
      dp = dextra;
      enddata += spill2;
      spill2 = enddata % 16;
      nwhole = enddata / 16;
      fseek(daf->fpr,foffset,SEEK_SET);
    } else {
      enddata = spill2;
      nwhole = 0;
    }
    for(j=0; j<nwhole; j++) {          /* encrypt and write out full blocks */
      aes_enc_blk(salt,buf,daf->ctx);
      for(i=0; i<16; i++)
        salt[i] = (*dp++) ^ buf[i];
      fwrite(salt,1,16,daf->fpr);
    }
    if(spill2 > 0) {                  /* deal with the end bit if there is one */
      aes_enc_blk(salt,buf,daf->ctx);
      for(i=0; i<spill2; i++)
        salt[i] = (*dp++) ^ buf[i];
      fwrite(salt,1,spill2,daf->fpr);
    }
    if(dextra) free(dextra);         /* free buffer if we used it */
  } else {
/*
  If not using encryption, things are easy: just find the correct place
  in the file and write the new data.  Write to the ifr file because
  that is the one SwitchFamilyMember sets: we are not normally expected to
  ever WRITE to any file other than the highest numbered one.
*/
    foffset = var->offset +var->ifile->FileLengthSize+
              var->ifile->FileCommandSize+var->ifile->FileTypeIDSize+
              strlen(var->name)+1+offset*tsizedisk;
    fseek(daf->fpr,foffset,SEEK_SET);
    retval = fwrite((octet *) data,tsizedisk,number,daf->fpr);
    if(retval < number) _errno = ERR_WRITE;
  }
  return retval;
}

int lsda_cd(int handle,char *path)
{
  LSDAFile *daf;
  int flag = 1;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_cd: invalid handle %d",handle);
    return -1;
  }
  daf = da_store + handle;

  if(daf->openmode == LSDA_READONLY) flag = 0;
  if(daf->ChangeDir(daf,path,flag) == NULL) {
    _errno = ERR_CD;
    if(report_level > 0) {
      if (daf->num_list > 0)
        fprintf(stderr,
        "lsda_cd: Cannot cd to %s in file %s.  Most likely a component of\nthe path is not a directory\n",
        path,daf->ifile[0]->filename);
      else
        fprintf(stderr, "lsda_cd: Cannot cd to %s.\n", path);
    }
    return -1;
  }
  daf->pathchanged=1;
  return 1;
}
STATIC int lsda_writecd(int handle,char *path)
{
  LSDAFile *daf = da_store + handle;
  Length rlen;
  int len;
  Command cmd = LSDA_CD;

  if(path == NULL) {
    daf->pathchanged=0;
    return 1;
  }
/*
  If were not finished with the previous variable close it out.
*/
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) {
      if(report_level > 0)
        fprintf(stderr,"lsda_writecd: error closing out variable\n");
      return -1;
    }
  }
  if(!daf->ifw->ateof) {
    fseek(daf->fpw,0,SEEK_END);
    daf->ifw->ateof = 1;
  }
  len = strlen(path);
  rlen = sizeof(Length)+sizeof(Command)+len;

  if(daf->encrypted && WriteSalt(daf)) goto write_error;
  if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto write_error;
  if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto write_error;
  if(WriteData((octet *) path,1,len,daf,1) < len) goto write_error;
  daf->pathchanged=0;
  daf->stpendlen += rlen;
  return 1;

write_error:
  _errno = ERR_WRITE;
  if(report_level > 0) {
    fprintf(stderr,
      "lsda_writecd: write error on file %s\n",fullfilename(daf->ifw));
  }
  return -1;
}

Length lsda_fsize(int handle)
{
  LSDAFile *daf;
  FILE *fp;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_fsize: invalid handle %d",handle);
    return 0;
  }
  _errno = ERR_NONE;

  daf = da_store+handle;
  fp = daf->fpw;

  if(fp == NULL) fp=daf->fpr;

  if(!fp) return 0;  /* no file currently opened */
  fseek(fp,0,SEEK_END);
  return((Length) ftell(fp)+daf->stpendlen);
}
int lsda_filenum(int handle)
/*
  This is only used to find the "end of file" so to speak -- the last
  file in the series that we are writing to.  It doesn't really make
  sense for files that have been opened READONLY
*/
{
  LSDAFile *daf;
  int ret;
  char *cp;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_filenum: invalid handle %d",handle);
    return -1;
  }
  daf = da_store+handle;

  if(daf->openmode == LSDA_READONLY) return 0;
  if(daf->ifw == NULL) return 0;
  cp = strrchr(daf->ifw->filename,'%');  /* find % */
  if(!cp) return 0;
  ret = atoi(cp+1);                          /* convert following to number */
  for(cp++; *cp; cp++)
    if(!isdigit(*cp)) ret=0;                 /* but only if all following are digits */
  return(ret);
}

int lsda_nextfile(int handle)
/*
  This also is only for writing -- to open the next file for writing.
*/
{
  LSDAFile *daf;
  IFile *ifile = NULL;
  int cur;
  char *cp,pwd[MAXPATH];

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_nextfile: invalid handle %d",handle);
    return -1;
  }
  daf = da_store+handle;

  if(daf->openmode == LSDA_READONLY) return 0;
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) goto cleanup;
  }
  if(daf->stpendlen && lsda_writesymboltable(daf) < 0) goto cleanup;
  if(daf->fpw && (daf->fpw != daf->fpr)) fclose(daf->fpw);
  daf->fpw=NULL;
  strcpy(pwd,daf->GetCWD(daf));
  daf->cwd = daf->top;

  cp = strrchr(daf->ifw->filename,'%');  /* find % */
  if(!cp) {
    cur=0;
  } else {
    cur = atoi(cp+1);                          /* convert following to number */
    for(cp++; *cp; cp++)
      if(!isdigit(*cp)) cur=0;                 /* but only if all following are digits */
  }
  /* Get new ifile to store stuff in */
  ifile = newIFile();
  ifile->dirname = (char *) malloc(strlen(daf->ifw->dirname)+1);
  strcpy(ifile->dirname,daf->ifw->dirname);
  ifile->filename = (char *) malloc(strlen(daf->ifw->filename)+5);
  strcpy(ifile->filename,daf->ifw->filename);
  /* Build new file name. */
  if(cur == 0) {
    strcat(ifile->filename,"%001");
  } else {
    cp = strrchr(ifile->filename,'%');
    sprintf(cp+1,"%3.3d",cur+1);
  }
  /*
    Store new Ifile in list
  */
  daf->ifile = (IFile **) realloc(daf->ifile,(daf->num_list+1)*sizeof(IFile *));
  daf->ifile[daf->num_list++] = ifile;
  daf->ifw = ifile;
  if((daf->fpw = fopen(fullfilename(ifile),"w+b"))!=NULL) {
    if(write_initialize(daf,NULL) < 0) goto cleanup;
  } else {
    _errno = ERR_OPENFILE;
    if(report_level > 0)
      fprintf(stderr,"lsda_nextfile: error opening file %s",fullfilename(daf->ifw));
    goto cleanup;
  }
  /*
    Go back to the same directory we were in
  */
  lsda_cd(handle,pwd);
  return cur+1;

cleanup:
  if(report_level > 0) fprintf(stderr,
    "lsda_nextfile: error processing file %s\n",fullfilename(daf->ifw));
  if(daf->fpw && (daf->fpw != daf->fpr)) fclose(daf->fpw);
  daf->fpw = NULL;
  if(daf->ifw) {
    if(daf->ifw->filename) free(daf->ifw->filename);
    if(daf->ifw->dirname) free(daf->ifw->dirname);
    free(daf->ifw);
  }
  daf->ifw = NULL;
  if(ifile) daf->ifile[--daf->num_list] = NULL;
  return -1;
}

int lsda_setmaxsize(int handle,Offset size)
/*
  Sets the handle's idea of the maximum allowable file size
*/
{
  LSDAFile *daf;
  Offset oldmax;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_setmaxsize: invalid handle %d",handle);
    return -1;
  }
  daf = da_store+handle;
  oldmax = daf->maxsize;
  if(size > DEF_MAX_SIZE)
    daf->maxsize = DEF_MAX_SIZE;
  else
    daf->maxsize = size;

  if(daf->fpw && daf->maxsize < oldmax) {
/*
  If we are getting too big, wrap to a new file
*/
    if(!daf->ifw->ateof) {
      fseek(daf->fpw,0,SEEK_END);
      daf->ifw->ateof = 1;
    }
    if(ftell(daf->fpw)+daf->stpendlen > daf->maxsize)
       lsda_nextfile(handle);
  }
  return 1;
}

int lsda_close(int handle)
{
  LSDAFile *daf = da_store+handle;

  if(handle < 0 || handle >= num_daf) goto cleanup;
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued && closeout_var(daf) < 0) goto cleanup;
  if(daf->stpendlen && lsda_writesymboltable(daf) < 0) goto cleanup;
  if(daf->fpw == daf->fpr) {
    if(daf->fpw) fclose(daf->fpw);
  } else {
    if(daf->fpr) fclose(daf->fpr);
    if(daf->fpw) fclose(daf->fpw);
  }
  daf->ifr= NULL;
  daf->fpr=NULL;
  daf->ifw= NULL;
  daf->fpw=NULL;
  daf->FreeTable(daf,daf->top);
  daf->FreeTypes(daf);
  if(daf->num_list) {
    int i;
    for(i=0; i<daf->num_list; i++) {
      if (daf->ifile[i]) {
        if (daf->ifile[i]->dirname)  free(daf->ifile[i]->dirname);
        if (daf->ifile[i]->filename) free(daf->ifile[i]->filename);
        free(daf->ifile[i]);
      }
    }
    free(daf->ifile);
  }
  daf->free = 1;
  return 1;

cleanup:
  if(report_level > 0) {
    if(daf->ifr)
      fprintf(stderr,"lsda_close: error closing file %s\n",fullfilename(daf->ifr));
    else if(daf->ifw)
      fprintf(stderr,"lsda_close: error closing file %s\n",fullfilename(daf->ifw));
  }
  _errno = ERR_CLOSE;
  return -1;
}

int lsda_flush(int handle)
{
  LSDAFile *daf = da_store+handle;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_flush: invalid handle %d",handle);
    return -1;
  }

  if(daf->openmode == LSDA_READONLY) return LSDA_SUCCESS;

  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) goto cleanup;
  }
  if(daf->stpendlen && lsda_writesymboltable(daf) < 0) goto cleanup;
  if(daf->fpw) {
    fflush(daf->fpw);
/*
  If we are getting too big, wrap to a new file
*/
    if(!daf->ifw->ateof) {
      fseek(daf->fpw,0,SEEK_END);
      daf->ifw->ateof = 1;
    }
    if(ftell(daf->fpw)+daf->stpendlen > daf->maxsize)
       lsda_nextfile(handle);
  }
  return LSDA_SUCCESS;
cleanup:
  if(report_level > 0) {
    fprintf(stderr,"lsda_flush: error processing file %s\n",
       fullfilename(daf->ifw));
  }
  return -1;
}
int lsda_sync(int handle)
{
  LSDAFile *daf = da_store+handle;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_sync: invalid handle %d",handle);
    return -1;
  }
  if(daf->openmode != LSDA_READONLY && daf->fpw) fflush(fileno(daf->fpw));//fsync(fileno(daf->fpw));
  return LSDA_SUCCESS;
}
LSDATable * LSDAresolve_link(LSDAFile *daf,LSDATable *varin)
{
  Offset foffset;
  char data[1024];
  LSDATable *var = varin;
  int k;
  char *pname, *newpname;
  pname = strdup(daf->GetPath(daf,var));   /* absolute path to var */

/*
  If file is write/read mode, flush stuff out to the file before we do a seek.
*/
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) return NULL;
  }
/*
  Loop reading link info and following it.
*/
  while(1) {

    if(!var->type || var->type->id != LSDA_LINK) return var;

    if(((daf->fpr == NULL) || (var->ifile != daf->ifr))
          && SwitchFamilyMember(daf,var) < 0) {
      if(report_level > 0) fprintf(stderr,"LSDAresolve_link: error\n");
      free(pname);
      return NULL;
    }
    daf->ifr->ateof = 0;
    foffset = var->offset +var->ifile->FileLengthSize+
              var->ifile->FileCommandSize+var->ifile->FileTypeIDSize+
              strlen(var->name)+1;
    /*
      If we have encryption, we have to read the salt from the 16
      bytes before the start of the data.
    */
    if(daf->encrypted) foffset += 16;
    fseek(daf->fpr,foffset,SEEK_SET);
    if(daf->encrypted) ReadSalt(daf);
    k=ReadData((unsigned char *)data,1,var->length,daf);

    if(k != var->length) {
      if(report_level > 0)
        fprintf(stderr,
          "LSDAresolve_link: error reading variable %s while reading file %s\n",
           pname,daf->ifr ? daf->ifr->filename : NULL);
      free(pname);
      return NULL;
    }
  /* need some kind of infinite loop protection.  Simplest would be
     to compare # times through here with the total size of the ST */
    data[var->length]=0;  /* file data is not NULL terminated */
    newpname = link_path(pname,data);
    free(pname);
    pname = newpname;
 /* don't let FindVar follow links, since that would call this routine and
    could lead to infinite recursion.
    But DO let it create directories, since we might be referencing dirs
    that were flushed to disk and no longer in the ST */
    var=daf->FindVar(daf,pname,1,0);
    if(var == NULL) {
      _errno = ERR_NOVAR;
      if(report_level > 0)
        fprintf(stderr,
          "LSDAresolve_link: variable %s not found while reading file %s\n",
           pname,daf->ifr ? daf->ifr->filename : NULL);
      free(pname);
      return NULL;
    }
  }
}

static Length lsda_realread(int handle, int type_id,char *name,Length offset,Length number,void *data,int follow)
{
  LSDAFile *daf;
  LSDAType *type;
  _CF Convert;
  int tsize;
  Offset foffset;
  Length ret;
  char buf[BUFSIZE], *cp;
  LSDATable *var;
  int tsizedisk;
  int k,kk,perbuf;
  Offset doff;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_read: invalid handle %d",handle);
    return -1;
  }
  if(number == 0) return 0;
  daf = da_store + handle;
  type = daf->FindTypeByID(daf,type_id);

  if(type == NULL) {
    _errno = ERR_DATATYPE;
    if(report_level > 0) {
      fprintf(stderr,"lsda_read: unrecognized data type %d",type_id);
      fprintf(stderr, " while reading file %s\n",daf->ifr ? daf->ifr->filename : NULL);
    }
    return -1;
  }
/*
  If file is write/read mode, flush stuff out to the file before we do a seek.
*/
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) return -1;
  }
/*
  get var info from symbol table: size of each item, starting offset
*/
  var=daf->FindVar(daf,name,0,follow);
  if(var == NULL) {
    _errno = ERR_NOVAR;
    if(report_level > 0)
      fprintf(stderr,
        "lsda_read: variable %s not found while reading file %s\n CWD=%s\n",
         name,daf->ifr ? daf->ifr->filename : NULL,daf->GetCWD(daf));
    return -1;
  }

  if(offset >= var->length) return 0;
  if(offset+number > var->length) number = var->length - offset;

  tsize = LSDASizeOf(type);
  tsizedisk = LSDASizeOf(var->type);
  if(daf->encrypted && (16 % tsizedisk)) {
  /*
    This cannot happen at the moment -- no data types are this large...
  */
    fprintf(stderr,"Error: reading varaible %s/%s from LSDA file %s:",
    daf->GetCWD(daf),name,daf->ifr->filename);
    fprintf(stderr,"       File is encrypted and variable size does not divide 16\n");
    _errno = ERR_READ;
    return 0;
  }
  Convert = GetConversionFunction(var->ifile,var->type,type);

  if(((daf->fpr == NULL) || (var->ifile != daf->ifr))
        && SwitchFamilyMember(daf,var) < 0) {
    if(report_level > 0) fprintf(stderr,"lsda_read: error\n");
    return -1;
  }
  daf->ifr->ateof = 0;
  cp=buf;
  /*
    If we have encryption, we can't just jump into the middle of
    the variable -- we have to read on 16 byte boundaries, and read
    the 16 bytes before the start of the data we want.
  */
  if(daf->encrypted) {
    int firstpad;  /* # bytes to throw out from first read */
    int64_t numwhole;      /* # whole reads to do */
    int lastreadlen;   /* # bytes to read for last block */
    int lastuse;       /* # bytes to use from last block */
    Offset fstart, fend, otmp;

    fstart = offset*tsizedisk;       /* Where data starts */
    fend = fstart+number*tsizedisk;  /* Where data ends */
    firstpad  = fstart % 16;     /* # extra bytes to read at start */
    otmp  = ((fend+15)/16)*16;  /* round end of data up to 16 byte boundary */
    if(otmp > var->length*tsizedisk) { /* but not past end of record */
      lastreadlen = 16-(otmp - var->length*tsizedisk);
    } else {
      lastreadlen = 16;
    }
    lastuse = 16-(otmp-fend);
    if(lastuse == 16) lastuse=0;
    numwhole=number*tsizedisk;
    if(firstpad) numwhole -= (16-firstpad);
    if(lastuse ) numwhole -= lastuse;
    numwhole /= 16;
    /*
       Add length of other stuff written at file location "var->offset"
       The 16 byte synchronization starts AFTER this stuff, so just
       add this in without any 16 byte boundary considerations.  Also
       note that this offset calculation does NOT include the 16 byte
       salt written at the head of the record, which is fine because
       we would just have to subtract it off since we want to back up
       16 bytes in the file anyway.  But it does include the 16 byte salt
       written just before the actual data.
    */
    foffset = fstart - firstpad;    /* Round down to 16 byte boundary */
    foffset += var->ifile->FileLengthSize+
               var->ifile->FileCommandSize+var->ifile->FileTypeIDSize+
               strlen(var->name)+1+16;
    fseek(daf->fpr,var->offset+foffset,SEEK_SET);
    ReadSalt(daf);
    /*
      Read first block
    */
    doff=0;
    if(firstpad) {
      if(numwhole < 0)  /* truncate data at both ends... */
        kk = (ReadData((unsigned char *)cp,1,lastreadlen,daf) - firstpad) / tsizedisk;
      else
        kk = (ReadData((unsigned char *)cp,1,16,daf) - firstpad) / tsizedisk;
      if(Convert) {
        Convert(cp+firstpad,(char *)data,kk);
      } else {
        memcpy((char *) data,cp+firstpad,kk*tsize);
      }
      doff=kk;
    }
    if(Convert) {
      int64_t i;
      perbuf = BUFSIZE / 16;  /* 16 byte blocks per buffer */
      k=perbuf;
      for(i=0; i<numwhole; i += perbuf) {
        if(i+k > numwhole) k=numwhole-i;
        kk = ReadData((unsigned char *)cp,1,16*k,daf) / tsizedisk;
        Convert(cp,((char *)data+doff*tsize),kk);
        doff += kk;
      }
    } else if(numwhole>0) {
      doff += ReadData((unsigned char *)data+doff*tsize,tsize,numwhole*16/tsize,daf);
    }
    if(lastuse && numwhole >= 0) {
      kk = ReadData((unsigned char *)cp,1,lastreadlen,daf);
      if(kk > lastuse) kk=lastuse;
      kk /= tsizedisk;
      if(Convert) {
        Convert(cp,(char *)data+doff*tsize,kk);
      } else {
        memcpy((char *) data+doff*tsize,cp,kk*tsize);
      }
      doff += kk;
    }
    ret=doff;
  } else {
    foffset = var->offset +var->ifile->FileLengthSize+
              var->ifile->FileCommandSize+var->ifile->FileTypeIDSize+
              strlen(var->name)+1+offset*tsizedisk;
/*
   Will this seek be a performance hit if we are in fact already at
   the correct file location?  I should hope not.  If this turns out
   to be the case, then maybe a call to ftell should be done first
   in case the caller is reading through some data in chunks
*/
    fseek(daf->fpr,foffset,SEEK_SET);
    if(Convert) {
    /*
      Read, in chunks, as many items as we can that will fit into
      our buffer, then convert them into the user's space
    */
      perbuf = BUFSIZE/tsizedisk;
      if(perbuf < 1) { /* Yoikes!  Big data item! */
        cp = (char *) malloc(tsizedisk);
        if(!cp) fprintf(stderr,"lsda_read: Malloc failed!\n"); exit(0);
        perbuf=1;
      }
      k=perbuf;
      ret=0;
      for(doff=0; doff<number; doff+= perbuf) {
        if(doff+k > number) k=number-doff;
        kk = fread(cp,tsizedisk,k,daf->fpr);
        Convert(cp,((char *)data)+doff*tsize,kk);
        ret=ret+kk;
        if(kk < k) break;
      }
      if(cp != buf) free(cp);
    } else {
      ret=ReadData((unsigned char *)data,tsize,number,daf);
    }
  }
  if(ret < number) {
    _errno = ERR_READ;
    if(report_level > 0) {
      fprintf(stderr,
        "lsda_read: error reading file %s\n",fullfilename(daf->ifr));
    }
  }
  return ret;
}
Length lsda_lread(int handle, int type_id,char *name,Length offset,Length number,void *data)
{
  return lsda_realread(handle,type_id,name,offset,number,data,0);
}
Length lsda_read(int handle, int type_id,char *name,Length offset,Length number,void *data)
{
  return lsda_realread(handle,type_id,name,offset,number,data,1);
}

STATIC int lsda_writesymboltable(LSDAFile *daf)
{
  Command cmd;
  Length rlen;
  Offset table_pos, cur_pos, offset_pos;
  char path1[MAXPATH],path2[MAXPATH];
/*
  If were not finished with the previous variable close it out.
*/
  if(daf->npend) WriteData(NULL,1,0,daf,1);
  if(daf->continued) {
    if(closeout_var(daf) < 0) goto cleanup1;
  }
  if(!daf->ifw->ateof) {
    fseek(daf->fpw,0,SEEK_END);
    daf->ifw->ateof = 1;
  }
  table_pos = ftell(daf->fpw);
  rlen = 0;
  if(daf->encrypted && WriteSalt(daf)) goto cleanup;
  if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto cleanup;
  cmd = LSDA_BEGINSYMBOLTABLE;
  if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto cleanup;
  if(daf->encrypted && WriteSalt(daf)) goto cleanup;  /* 1 salt for all symbols */

  path1[0] = path2[0] = 0;
  if(lsda_writesymbol(path1,path2,daf->top,daf) < 0) goto cleanup1;

/*
  Write end of symbol table record
*/
  rlen = sizeof(Length)+sizeof(Command)+sizeof(Offset);
  if(daf->encrypted && WriteSalt(daf)) goto cleanup;
  if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto cleanup;
  cmd = LSDA_ENDSYMBOLTABLE;
  if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto cleanup;
  offset_pos = ftell(daf->fpw);
  cur_pos = 0;  /* offset to next piece of table -- 0=> no next piece */
  if(WriteData((octet *) &cur_pos,sizeof(Offset),1,daf,1) < 1) goto cleanup;
/*
  Update length of this symbol table block
*/
  cur_pos = ftell(daf->fpw);
  rlen = cur_pos-table_pos;
  daf->ifw->ateof = 0;
  fseek(daf->fpw,table_pos,SEEK_SET);
  if(daf->encrypted) {  /* have to rewrite the whole record */
    if(WriteSalt(daf)) goto cleanup;
    rlen -= 16;  /* "table_pos" included leading SALT, which isn't properly
                    inside of the record, so don't count it in the length */
    if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto cleanup;
    cmd = LSDA_BEGINSYMBOLTABLE;
    if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto cleanup;
  } else {
    if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto cleanup;
  }
/*
  Update the file offset that points to this chunk of the symbol table
*/
  if(daf->encrypted) {  /* have to rewrite the whole record */
    fseek(daf->fpw,daf->ifw->stoffset-sizeof(Length)-sizeof(Command) - 16,SEEK_SET);
    if(WriteSalt(daf)) goto cleanup;
    rlen = sizeof(Length)+sizeof(Command)+sizeof(Offset);
    if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto cleanup;
    cmd = LSDA_ENDSYMBOLTABLE;
    if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto cleanup;
    if(WriteData((octet *) &table_pos,sizeof(Offset),1,daf,1) < 1) goto cleanup;
  } else {
    fseek(daf->fpw,daf->ifw->stoffset,SEEK_SET);
    if(WriteData((octet *) &table_pos,sizeof(Offset),1,daf,1) < 1) goto cleanup;
  }
  daf->ifw->stoffset = offset_pos;
  daf->stpendlen = 0;
/*
  If file is not being read, delete unneeded symbols
*/
  if(daf->openmode == LSDA_WRITEONLY) PruneSymbols(daf,daf->top);
  return 1;

cleanup1:
  if(report_level > 0) fprintf(stderr,
    "lsda_writesymboltable: error processing file %s\n",fullfilename(daf->ifw));
  return -1;

cleanup:
  _errno = ERR_WRITE;
  if(report_level > 0) {
    fprintf(stderr,
        "lsda_writesymboltable: write error on file %s\n",
        fullfilename(daf->ifw));
  }
  return -1;
}

STATIC void PruneSymbols(LSDAFile *daf,LSDATable *symbol)
{
  int i;
  LSDATable **kids;
  int numkids;

  if(symbol->type) {    /* a variable */
    if(!symbol->dirty) {
      daf->FreeTable(daf,symbol);
      return;
    }
  } else {
    if(symbol->children) {
      numkids = BT_numentries(symbol->children);
      if(numkids) {
        kids = (LSDATable **) BT_list(symbol->children);
        for(i=0; i<numkids; i++)
          PruneSymbols(daf,kids[i]);
        free(kids);
      }
    }
    if(!symbol->children || BT_numentries(symbol->children) == 0) {
      if(symbol != daf->top && symbol != daf->cwd)
        daf->FreeTable(daf,symbol);
    }
  }
}
size_t lsda_totalmemory(int handle)
{
  LSDAFile *daf = da_store + handle;
  if(!daf->top) return (size_t) 0;
  return SymbolSizes(daf,daf->top);
}

STATIC size_t SymbolSizes(LSDAFile *daf,LSDATable *symbol)
{
  int cont;
  LSDATable *child;
  size_t tot = 0;

  if(symbol->type) {    /* a variable */
    tot = symbol->length*LSDASizeOf(symbol->type);
  } else {
    if(symbol->children) {
      for(cont=0; ; ) {
       child = (LSDATable *) BT_enumerate(symbol->children,&cont);
       if(!child) break;
       tot += SymbolSizes(daf,child);
      }
    }
  }
  return tot;
}

STATIC int lsda_readsymboltable(LSDAFile *daf)
{
  Command cmd;
  Offset table_pos, offset_pos, end_pos;
  Length rlen;
/*
  Read symbol table from file.
*/
  if(fseek(daf->fpr,daf->ifr->stoffset,SEEK_SET) < 0) {
    _errno = ERR_FSEEK;
    goto cleanup;
  }
  table_pos = ReadOffset(daf);
  if(_errno != ERR_NONE) goto cleanup;

  while(table_pos) {
    if(fseek(daf->fpr,table_pos,SEEK_SET) < 0) {
      _errno = ERR_FSEEK;
      goto cleanup;
    }
    if(daf->encrypted && ReadSalt(daf)) goto cleanup;
    rlen = ReadLength(daf);
    if(_errno != ERR_NONE) goto cleanup;
    end_pos = table_pos+rlen-sizeof(Length)-sizeof(Command)-sizeof(Offset);

    cmd = ReadCommand(daf);
    if(_errno != ERR_NONE) goto cleanup;
    if(cmd != LSDA_BEGINSYMBOLTABLE) {
      _errno = ERR_NOBEGINSYMBOLTABLE;
      goto cleanup;
    }
    if(daf->encrypted && ReadSalt(daf)) goto cleanup;  /* 1 salt for all symbols */
    while(ftell(daf->fpr) < end_pos && lsda_readsymbol(daf))
      ;
/*
  Check for end of symbol table record
*/
    if(daf->encrypted && ReadSalt(daf)) goto cleanup;
    ReadLength(daf);
    if(_errno != ERR_NONE) goto cleanup;
    cmd = ReadCommand(daf);
    if(_errno != ERR_NONE) goto cleanup;
    if(cmd != LSDA_ENDSYMBOLTABLE) {
      _errno = ERR_NOENDSYMBOLTABLE;
      goto cleanup;
    }
    offset_pos = ftell(daf->fpr);
    table_pos = ReadOffset(daf);
    if(_errno != ERR_NONE) goto cleanup;
  }
  daf->ifr->stoffset = offset_pos;
  daf->ifr->ateof = 0;
  return 1;

cleanup:
  if(report_level > 0)
     fprintf(stderr,
      "lsda_readsymboltable: error %d on file %s at byte %ld\n",
      _errno,fullfilename(daf->ifr),(long) ftell(daf->fpr));
  return -1;
}
STATIC int lsda_writesymbol(char *ppath,char *curpath,
                           LSDATable *symbol,LSDAFile *daf)
{
  Command cmd;
  Length rlen;
  int nlen;
  char *pp;
  int pplen;
  int i,keeplen,cont;
  int retval = 0;
  LSDATable *child;

  if(!symbol->dirty) return 0;
  if(symbol->type) {  /* dirty variable */
    nlen = strlen(symbol->name);
    if(strcmp(ppath,curpath)) { /* have to write a directory entry */
      pp = findpath(ppath,curpath);
      pplen = strlen(pp);
      rlen = pplen+sizeof(Length)+sizeof(Command);
      if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto cleanup;
      cmd = LSDA_CD;
      if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto cleanup;
      if(WriteData((octet *) pp,1,pplen,daf,1) < pplen) goto cleanup;
      strcpy(ppath,curpath);
    }
    rlen = sizeof(Length)+sizeof(Command)+sizeof(TypeID)+nlen+
           sizeof(Offset)+sizeof(Length);
    if(WriteData((octet *) &rlen,sizeof(Length),1,daf,1) < 1) goto cleanup;
    cmd = LSDA_VARIABLE;
    if(WriteData((octet *) &cmd,sizeof(Command),1,daf,1) < 1) goto cleanup;
    if(WriteData((octet *) symbol->name,1,nlen,daf,1) < nlen) goto cleanup;
    if(WriteData((octet *) &symbol->type->id,sizeof(TypeID),1,daf,1) < 1) goto cleanup;
    if(WriteData((octet *) &symbol->offset,sizeof(Offset),1,daf,1) < 1) goto cleanup;
    if(WriteData((octet *) &symbol->length,sizeof(Length),1,daf,1) < 1) goto cleanup;
    retval = 1;
  } else if(symbol->children) {   /* do subdir */
    keeplen = strlen(curpath);
    if(keeplen == 0)
      strcpy(curpath,"/");
    else if(keeplen == 1)
      sprintf(curpath+keeplen,"%s",symbol->name);
    else
      sprintf(curpath+keeplen,"/%s",symbol->name);
/*
  Go through two times:  First write any simple variables,
  then do subdirectories
*/
    for(cont=0; ;) {
      child = (LSDATable *) BT_enumerate(symbol->children,&cont);
      if(!child) break;
      if(child->type) {
        i=lsda_writesymbol(ppath,curpath,child,daf);
        if(i < 0) {
          if(report_level > 0) fprintf(stderr,"lsda_writesymbol: error\n");
          return -1;
        }
      }
    }
    for(cont=0; ;) {
      child = (LSDATable *) BT_enumerate(symbol->children,&cont);
      if(!child) break;
      if(!child->type) {
        i=lsda_writesymbol(ppath,curpath,child,daf);
        if(i < 0) {
          if(report_level > 0) fprintf(stderr,"lsda_writesymbol: error\n");
          return -1;
        }
      }
    }
    retval = 1;
    curpath[keeplen]=0;
  }
  symbol->dirty = 0;
  return(retval);

cleanup:
  _errno = ERR_WRITE;
  if(report_level > 0) {
    fprintf(stderr,
      "lsda_writesymbol: write error on file %s",fullfilename(daf->ifw));
  }
  return -1;
}
STATIC LSDATable *lsda_readsymbol(LSDAFile *daf)
{
  Command cmd;
  Length rlen;
  int nlen;
  char name[256];
  TypeID type_id;
  LSDATable *symbol;
  LSDAType *type;
  
top:
  rlen=ReadLength(daf);
  if(_errno == ERR_READ) goto cleanup;
  cmd=ReadCommand(daf);
  if(_errno == ERR_READ) goto cleanup;
  if(cmd == LSDA_VARIABLE) {
    nlen = rlen-
         2*daf->ifr->FileLengthSize-daf->ifr->FileCommandSize-
           daf->ifr->FileTypeIDSize-daf->ifr->FileOffsetSize;
    if(ReadData((unsigned char *)name,1,nlen,daf) < nlen) {
      _errno = ERR_READ;
      goto cleanup;
    }
    name[nlen] = 0;
    type_id = ReadTypeID(daf);
    if(_errno == ERR_READ) goto cleanup;
    type = daf->FindTypeByID(daf,type_id);
    if(type == NULL) {
      _errno = ERR_NOTYPEID;
      if(report_level > 0) {
        fprintf(stderr,
          "lsda_readsymbol: No corresponding id for %d in file %s\n",
          (int) type_id,fullfilename(daf->ifr));
      }
      return NULL;
    }
    symbol = daf->CreateVar(daf,type,name);
    symbol->offset = ReadOffset(daf);  
    if(_errno == ERR_READ) goto cleanup;
    symbol->length = ReadLength(daf);
    if(_errno == ERR_READ) goto cleanup;
    symbol->ifile = daf->ifr;
  } else if(cmd == LSDA_CD) {
    nlen = rlen-daf->ifr->FileLengthSize-daf->ifr->FileCommandSize;
    if(ReadData((unsigned char *)name,1,nlen,daf) < nlen) {
      _errno = ERR_READ;
      goto cleanup;
    }
    name[nlen]=0;
    if(daf->ChangeDir(daf,name,1) == NULL) {
      _errno = ERR_CD;
      if(report_level > 0) {
        fprintf(stderr,"lsda_readsymbol: Cannot cd to %s in file %s\n",
                name,fullfilename(daf->ifr));
        fprintf(stderr,"Most likely a component of\nthe path is not a directory\n");
      }
      return NULL;
    }
    symbol = daf->cwd;
  } else if(cmd == LSDA_NULL) {  /* ignore NULL commands */
    nlen = rlen-daf->ifr->FileLengthSize-daf->ifr->FileCommandSize;
    fseek(daf->fpr,nlen,SEEK_CUR);
    goto top;
  } else {
    fseek(daf->fpr,-daf->ifr->FileLengthSize-daf->ifr->FileCommandSize,SEEK_CUR);
    return NULL;
  }
  return symbol;

cleanup:
  if(report_level > 0) {
      fprintf(stderr,
      "lsda_readsymbol: read error on file %s",fullfilename(daf->ifr));
  }
  return NULL;
}
#ifdef DUMP_DEBUG
lsda_dumpst(int handle)
{
  LSDAFile *daf = da_store + handle;
  dumpit("/",daf->top);
}
dumpit(char *cwd, LSDATable *symbol)
{
  char dir[1024];
  int cont;
  LSDATable *child;

  if(symbol->type) {
    printf("Var %s%s, type = %s,  file %s, offset = %d, length = %d\n",cwd,
      symbol->name,symbol->type->name,symbol->ifile->filename, symbol->offset,symbol->length);
  } else {
    if(strcmp(cwd,"/")==0 && strcmp(symbol->name,"/") == 0) {
        strcpy(dir,"/");
    } else {
      sprintf(dir,"%s%s/",cwd,symbol->name);
    }
    printf("Dir %s\n",dir);
    if(symbol->children) {
      for(cont=0; ;) {
        child = (LSDATable *) BT_enumerate(symbol->children,&cont);
        if(!child) break;
        dumpit(dir,child);
      }
    }
  }
}
#endif

STATIC Length ReadLength(LSDAFile *daf)
{
  return * (Length *) ReadTrans(daf,daf->ifr->FileLengthSize,daf->ifr->ConvertLength);
}
STATIC Offset ReadOffset(LSDAFile *daf)
{
  return * (Offset *) ReadTrans(daf,daf->ifr->FileOffsetSize,daf->ifr->ConvertOffset);
}
STATIC Command ReadCommand(LSDAFile *daf)
{
  return * (Command *) ReadTrans(daf,daf->ifr->FileCommandSize,daf->ifr->ConvertCommand);
}
STATIC TypeID ReadTypeID(LSDAFile *daf)
{
  return * (TypeID *) ReadTrans(daf,daf->ifr->FileTypeIDSize,daf->ifr->ConvertTypeID);
}
STATIC int ReadSalt(LSDAFile *daf)
{
/*
  Read salt from the file... pretty straight forward.
*/
  if(fread(daf->salt, 1, 16, daf->fpr) != 16) return 1;
  return 0;
}
STATIC Length ReadData(octet *data, size_t size, size_t count, LSDAFile *daf)
{
  size_t bytes = size*count;
  octet buf[16];
  Length doff;
  int i;

  if(daf->encrypted) {
/*
  Decrypt data using CFB mode
*/
    doff=0;
    while(bytes >= 16) {
      aes_enc_blk((char *)daf->salt,(char *)buf,daf->ctx);
      i=fread(daf->salt, 1, 16, daf->fpr);
      if(i < 16) return (doff/size);
      for(i=0; i<16; i++)
        data[i] = daf->salt[i] ^ buf[i];
      doff += 16;
      data += 16;
      bytes -= 16;
    }
    if(bytes > 0) {
      aes_enc_blk((char *)daf->salt,(char *)buf,daf->ctx);
      for(i=bytes; i<16; i++)
        daf->salt[i-bytes] = daf->salt[i];
      doff += fread(daf->salt+16-bytes, 1, bytes, daf->fpr);
      for(i=0; i<bytes; i++)
        data[i] = daf->salt[16-bytes+i] ^ buf[i];
    }
    return (doff/size);
  } else {
    return fread(data, size, count, daf->fpr);
  }
}
STATIC int WriteSalt(LSDAFile *daf)
{
/*
  Encrypt salt to increment it, so it will not be whatever was
  used for the last encryption.  Then write it to the file, which
  is equivalent to encrypting 16 bytes of 0 and writing that to the
  file in CFB mode.  The point is that we need to write out the salt at the
  beginning of each record so we have it to synchronize decryption.
*/
  aes_enc_blk((char *)daf->salt,(char *)daf->salt,daf->ctx);
  if(fwrite(daf->salt, 1, 16, daf->fpw) != 16) return 1;
  return 0;
}
STATIC Length WriteData(octet *data, size_t size, size_t count, LSDAFile *daf,int flush)
{
  size_t bytes = size*count;
  octet buf[16];
  int i,j,k;
  Length doff;

  if(daf->encrypted) {
/*
  Encrypt data using CFB mode

  Copy data into pending buffer until buffer is full or we
  run out of data...
*/
    for(doff=0; doff<bytes && daf->npend < 16; doff++)
      daf->pending[daf->npend++] = *data++;
    bytes -= doff;
/*
  If pending buffer is full, write it out
*/
    if(daf->npend == 16) {
      aes_enc_blk((char *)daf->salt,(char* ) buf,daf->ctx);
      for(i=0; i<16; i++)
        daf->salt[i] = daf->pending[i] ^ buf[i];
      k=fwrite(daf->salt, 1, 16, daf->fpw);
      if(k < 16) return ((doff+k-16)/size);
      daf->npend=0;
    }
/*
  If there is more data to write, write as many 16 byte
  chunks as we can
*/
    while(bytes >= 16) {
      aes_enc_blk((char *)daf->salt,(char *)buf,daf->ctx);
      for(i=0; i<16; i++)
        daf->salt[i] = data[i] ^ buf[i];
      k=fwrite(daf->salt, 1, 16, daf->fpw);
      if(k < 16) return ((doff+k)/size);
      doff += 16;
      data += 16;
      bytes -= 16;
    }
/*
  If there is data left, put it in the pending buffer.
  If there is data, the pending buffer must be empty, because
  we would have flushed it above
*/
    if(bytes > 0) {
      for(i=0; i<bytes ; i++)
        daf->pending[daf->npend++] = *data++;
      doff+=bytes;
    }
/*
  Finally, flush the pending buffer if requested
*/
    if(flush && daf->npend) {
      aes_enc_blk((char *)daf->salt,(char *)buf,daf->ctx);
      for(i=0; i<daf->npend; i++)
        buf[i] ^= daf->pending[i];
      k=fwrite(buf, 1, daf->npend, daf->fpw);
      for(i=daf->npend; i<16; i++)
        daf->salt[i-daf->npend] = daf->salt[i];
      for(i=0; i<daf->npend; i++)
        daf->salt[16-daf->npend+i] = buf[i];
      k = daf->npend - k;   /* # bytes short on write */
      daf->npend=0;
      if(k) return (count-(k+size-1)/size);
    }
    return (doff/size);
  } else {
    return fwrite(data, size, count, daf->fpw);
  }
}
STATIC void *ReadTrans(LSDAFile *daf,int FileLength,_CF Convert)
{
  static char buf[16],buf2[16];

  if(ReadData((unsigned char *)buf,1,FileLength,daf) < 1) {
    memset(buf,0,16);
    _errno = ERR_READ;
    if(report_level > 0) {
      fprintf(stderr,
          "ReadTrans: error reading %d bytes from file %s\n",
          FileLength,fullfilename(daf->ifr));
    }
    return (void *) buf;
  }
  if(Convert) {
    Convert(buf,buf2,1);
    return (void *) buf2;
  } else {
    return (void *) buf;
  }
}
STATIC char *findpath(char *from, char *to)
{
int i,j,k;
int lastdir;
int lento = strlen(to);
int lenrel;
static char relpath[256];


lastdir=0;
if(from[0]==0 ) return to;            /* we HAVE no previous path.... */
if(from[1]==0 ) return to;            /* "from" is root dir */
for(i=0; from[i] && from[i] == to[i]; i++)  /* find common part of path */
  if(from[i] == '/') lastdir=i;
if(from[i] == '/' && !to[i]) lastdir=i;

if(!from[i] && !to[i]) return NULL;   /* identical */


if(!from[i] && to[i] == '/') return to+i+1;  /* to is a subdir of from */

/*
  Count number of ".." we'd need to do a relative path
*/

for(j=lastdir,k=0; from[j]; j++)
  if(from[j] == '/') k++;

lenrel = 3*k + (lento-lastdir);

if(lenrel < lento) {
  for(i=0; i<3*k; i+=3) {
    relpath[i  ] = '.';
    relpath[i+1] = '.';
    relpath[i+2] = '/';
  }
  if(lento == lastdir)   /* to is a subdir of from */
    relpath[i-1]=0;      /* strip off last / */
  else
    sprintf(relpath+i,"%s",to+lastdir+1);
  return relpath;
} else
  return to;
}

/*
STATIC char *finddirmatch(char *name,DIR *dp)
{
  struct dirent *file;
  int i,nlen;
  char *cp;

again:
  while(file = readdir(dp)) {

//  return this file if it is of the form "name%XXXX"

    return file->d_name;

    nlen = strlen(name);
    cp = file->d_name;
    for(i=0; i<nlen; i++,cp++)
      if(*cp != name[i]) goto again;
    if(*cp == 0) return name;
    if(*cp != '%') goto again;
    for(cp++; *cp; cp++)
       if(!isdigit(*cp)) goto again;
    return file->d_name;
  }
  return NULL;
}
*/


STATIC char *finddirmatch(char *name,DIR *dp)
{
  struct dirent *file;
  int i,nlen;
  char *cp;

again:
  while(file = readdir(dp)) {

//  return this file if it is of the form "name%XXXX"

    return file->d_name;

    nlen = strlen(name);
    cp = file->d_name;
    for(i=0; i<nlen; i++,cp++)
      if(*cp != name[i]) goto again;
    if(*cp == 0) return name;
    if(*cp != '%') goto again;
    for(cp++; *cp; cp++)
       if(!isdigit(*cp)) goto again;
    return file->d_name;
  }
  return NULL;
}


LSDADir * lsda_opendir(int handle,char *path)
{
  LSDAFile *daf;
  LSDATable *var;
  LSDADir *dir;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_opendir: invalid handle %d",handle);
    return NULL;
  }
  daf = da_store + handle;

  var=daf->FindVar(daf,path,0,1);
  if(var == NULL || var->type) {
    _errno = ERR_OPNDIR;
    if(report_level > 0) 
      fprintf(stderr,
    "lsda_opendir: cannot find directory %s in file %s%c%s",path,
    daf->ifr->dirname,DIR_SEP,daf->ifr->filename);
    return NULL;
  }
  dir = (LSDADir *) malloc(sizeof(LSDADir));
  dir->btree = var->children;
  dir->cont = 0;
  dir->daf = (void *) daf;
  return dir;
}
void lsda_readdir(LSDADir *dir,char *name,int *type_id,Length *length,
                int *filenum)
{
  LSDATable *var;

  if(!dir || !dir->btree) { /* they forgot to call opendir, or dir is empty */
    name[0]=0;
    *type_id = -1;
    *length = *filenum = -1;
    return;
  }
  var = (LSDATable *) BT_enumerate(dir->btree,&dir->cont);
  if(var) {
    strcpy(name,var->name);
    if(var->type) {
      *type_id = LSDAId(var->type);
      *length = var->length;
      *filenum = 0;  /* obsolete */
    } else {
      *type_id = 0;
      if(var->children)
        *length = BT_numentries(var->children);
      else
        *length = 0;
      *filenum = -1;
    }
  } else {
    name[0]=0;
    *type_id = -1;
    *length = *filenum = -1;
  }
}
void lsda_closedir(LSDADir *dir)
{
  if(dir) free(dir);
}
void lsda_realquery(int handle,char *name,int *type_id,Length *length,int follow)
{
  LSDAFile *daf;
  LSDATable *var;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    *type_id= -1;
    *length = 0;
    if(report_level > 0) fprintf(stderr, "lsda_query: invalid handle %d",handle);
    return;
  }

  daf = da_store + handle;

  var=daf->FindVar(daf,name,0,follow);
  if(var) {
    if(var->type) {
      *type_id = LSDAId(var->type);
      *length = var->length;
    } else {
      *type_id= 0;
      if(var->children)
        *length = BT_numentries(var->children);
      else
        *length = 0;
    }
  } else {
    *type_id= -1;
    *length = 0;
  }
}
void lsda_query(int handle,char *name,int *type_id,Length *length)
{
  lsda_realquery(handle,name,type_id,length,1);
}
void lsda_lquery(int handle,char *name,int *type_id,Length *length)
{
  lsda_realquery(handle,name,type_id,length,0);
}

void lsda_queryvar(int handle,char *name,int *type_id,Length *length,
                int *filenum)
{
  LSDAFile *daf;
  LSDATable *var;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    *type_id= -1;
    *length = 0;
    *filenum = -1;
    if(report_level > 0) fprintf(stderr, "lsda_queryvar: invalid file handle %d",handle);
    return;
  }

  daf = da_store + handle;
  var=daf->FindVar(daf,name,0,1);
  if(var) {
    if(var->type) {
      *type_id = LSDAId(var->type);
      *length = var->length;
      *filenum = 0;  /* obsolete */
/*
      *offset = var->offset +daf->FileLengthSize+daf->FileCommandSize+
            daf->FileTypeIDSize + strlen(var->name)+1;
*/
    } else {
      *type_id= 0;
      if(var->children)
        *length = BT_numentries(var->children);
      else
        *length = 0;
      *filenum = -1;
    }
  } else {
    *type_id= -1;
    *length = 0;
    *filenum = -1;
  }
}

#define MAX_BUFSIZE 10485760   /* 10 MB max buffer size */
int lsda_copydir(int h1, char *dir1, int h2, char *dir2)
{
 /* Recursively copy the contents of dir1 in file h1 into
    dir2 in file h2
 */
  char name[MAXPATH],name2[MAXPATH];
  int type, fno,bper;
  Length btot,dsize;
  Length length,offset;
  unsigned char *data;
  LSDADir *dir;
  static int bufsize=0, depth=0;
  static unsigned char *buf=NULL;
  int rc;
  
  depth++; /* keep track of recursion depth */

  if(buf==NULL) {  /* get read/write buffer */
    bufsize=65536;
    buf = (unsigned char *) malloc(bufsize);
  }
  rc= 1;  /* default return code is "error" */
  
  if(lsda_cd(h1, dir1)<0 || lsda_cd(h2, dir2)<0) goto done;
  dir = lsda_opendir(h1, ".");
  if(dir==NULL) goto done;

  do {
    lsda_readdir(dir, name, &type, &length, &fno);  /* get next entry */
    if(type == 0) {
      if(lsda_copydir(h1,name,h2,name)) goto done;  /* subdir -- recur */
      lsda_cd(h1,"..");    /* return to my directory */
      lsda_cd(h2,"..");
    } else if(type > 0) {  /* variable */
      bper = lsda_util_id2size(type);
      btot = bper*length;  /* total # bytes */
      if(btot > bufsize && bufsize < MAX_BUFSIZE) {   /* is buffer big enough? */
        bufsize = btot < MAX_BUFSIZE ? btot : MAX_BUFSIZE;
        free(buf);
        buf = (unsigned char *) malloc(bufsize);
      }
      dsize = bufsize/bper;           /* entries per buffer */
      offset=0;
      strcpy(name2,name);
      do {               /* read/write in buffer size chunks */
        if(dsize > length-offset) dsize=length-offset;
        lsda_read(h1,type,name,offset,dsize,buf);
        lsda_write(h2,type,name2,dsize,buf);
        offset=offset+dsize;
        name2[0]=0;         /* reset name for "continuation" */
      } while (offset < length);
    }
  } while(type >= 0);
  lsda_closedir(dir);
  rc=0;    /* Success */
done:
  depth--;
  if(depth == 0) {  /* if done, free buffer */
    free(buf);
    buf=NULL;
    bufsize=0;
    lsda_flush(h2);  /* flush receiving file */
  }
  return rc;
}

char *lsda_getpwd(int handle)
{
  LSDAFile *daf;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    _scbuf[0]=0;
    if(report_level > 0) fprintf(stderr, "lsda_getpwd: invalid handle %d",handle);
    return _scbuf;
  }

  daf = da_store + handle;
  return daf->GetCWD(daf);
}
void lsda_setreportlevel(int level)
{
  report_level = level;
}
int lsda_nextopen(int handle)
{
int i;

if(handle < 0) handle = -1;

for(i=handle+1; i<num_daf; i++)
   if(da_store[i].free == 0) return i;

return -1;
}
static char *fullfilename(IFile *ifp)
{
  sprintf(_scbuf,"%s%c%s",ifp->dirname,DIR_SEP,ifp->filename);
  return _scbuf;
}
void lsda_perror(char *string)
{
  fprintf(stderr,"%s : ",string);
  switch (_errno) {
    case ERR_NONE:              /* no error */
       fprintf(stderr,"No error\n");
       break;
    case ERR_MALLOC:            /* malloc failed */
       fprintf(stderr,"Malloc failed\n");
       break;
    case ERR_NOFILE:            /* non-existent file */
       fprintf(stderr,"Attempt to reopen non-existant file\n");
       break;
    case ERR_FSEEK:             /* fseek failed */
       fprintf(stderr,"Fseek failed\n");
       break;
    case ERR_READ:              /* read error on file */
       fprintf(stderr,"Read error\n");
       break;
    case ERR_WRITE:             /* write error on file */
       fprintf(stderr,"Write error\n");
       break;
    case ERR_NOENDSYMBOLTABLE:  /* append, but end of symbol table not found */
       fprintf(stderr,"Attempt to truncate file at invalid location\n");
       break;
    case ERR_OPENDIR:           /* error opening directory for file */
       fprintf(stderr,"Error opening directory for file operation\n");
       break;
    case ERR_OPENFILE:          /* error opening file */
       fprintf(stderr,"Error opening file\n");
       break;
    case ERR_NOCONT:            /* empty name to write when not continuing */
       fprintf(stderr,"Write with empty variable name when\n");
       fprintf(stderr,"last file operation was not a write\n");
       break;
    case ERR_DATATYPE:          /* write with unknown data type */
       fprintf(stderr,"Write attempt with unknown variable type\n");
       break;
    case ERR_NOTYPEID:          /* read unknown type id from file */
       fprintf(stderr,"Read unknown type id from file\n");
       break;
    case ERR_CD:                /* illegal cd attempt in file */
       fprintf(stderr,"Illegal directory change\n");
       fprintf(stderr,"Most likely a component in the specified path");
       fprintf(stderr,"already exists as a\n variable\n");
       break;
    case ERR_CLOSE:             /* error on close ?? */
       fprintf(stderr,"Error closing file\n");
       break;
    case ERR_NOVAR:             /* read on non-existant variable */
       fprintf(stderr,"Attempt to read on non-existant variable\n");
       break;
    case ERR_NOBEGINSYMBOLTABLE:/* missing Begin Symbol Table */
       fprintf(stderr,"Error: missing BEGINSYMBOLTABLE\n");
       break;
    case ERR_OPNDIR:            /* open directory in file for query */
       fprintf(stderr,"Error opening directory for query\n");
       break;
    default:
       fprintf(stderr,"Unknown error %d\n",_errno);
       break;
  }
  _errno = ERR_NONE;
}
char *lsda_getname(int handle)
{
  LSDAFile *daf;
  char *cp, *c;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    _scbuf[0]=0;
    if(report_level > 0) fprintf(stderr, "lsda_getname: invalid handle %d",handle);
    return _scbuf;
  }
  daf = da_store + handle;

  if(!daf->ifr && daf->ifile) daf->ifr = daf->ifile[0];
  if(!daf->ifr) { _scbuf[0]=0; return _scbuf; }
  sprintf(_scbuf,"%s%c%s",daf->ifr->dirname,DIR_SEP,daf->ifr->filename);
  cp = strrchr(_scbuf,'%');  /* strip out the part after % */
  if(cp) {
    for(c=cp+1; *c; c++)
      if(!isdigit(*c)) cp=NULL;
  }
  if(cp) *cp=0;
  return _scbuf;
}

char *lsda_getbasename(int handle)
{
  LSDAFile *daf;
  char *cp, *c;

  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    _scbuf[0]=0;
    if(report_level > 0) fprintf(stderr, "lsda_getbasename: invalid handle %d",handle);
    return _scbuf;
  }
  daf = da_store + handle;

  if(!daf->ifr && daf->ifile) daf->ifr = daf->ifile[0];
  if(!daf->ifr) { _scbuf[0]=0; return _scbuf; }
  strcpy(_scbuf,daf->ifr->filename);
  cp = strrchr(_scbuf,'%');  /* strip out the part after % */
  if(cp) {
    for(c=cp+1; *c; c++)
      if(!isdigit(*c)) cp=NULL;
  }
  if(cp) *cp=0;
  return _scbuf;
}

int lsda_getmode(int handle)
{
  if(handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    if(report_level > 0) fprintf(stderr, "lsda_getmode: invalid handle %d",handle);
    return -1;
  }
  return da_store[handle].openmode;
}

int lsda_gethandle(char *filen)
{
int i,len;
LSDAFile *daf;
/*
  Scan open handles and check for one matching the given name
*/
_errno = ERR_NONE;  /* reset error */
len = strlen(filen);

for(i=0; i<num_daf; i++) {
  if(!da_store[i].free) {
    daf = da_store+i;
    if(!daf->ifr && daf->ifile) daf->ifr = daf->ifile[0];
    if(!daf->ifr) continue;
    strcpy(_scbuf,daf->ifr->filename);
    if(_scbuf[len] == '%') _scbuf[len]=0;
    if(strcmp(_scbuf,filen)==0) return i;
  }
}
return -1;
}

int lsda_util_countdir(int fhandle, char * dirname, int *ndir)
{
  LSDADir *dir;
  char childdirname[80];
  int tid, fno;
  size_t len;

  if(lsda_cd(fhandle, dirname)<0) return -1;
  dir = lsda_opendir(fhandle, ".");
  if(dir==NULL) return -1;
  do {
    lsda_readdir(dir, childdirname, &tid, &len, &fno);
    if(childdirname[0]) (*ndir)++;
  } while(childdirname[0]);
  lsda_closedir(dir);
  return *ndir;
}

int lsda_util_id2kind(int type_id)
{
  switch(type_id)
  {
  case LSDA_I1:
  case LSDA_I2:
  case LSDA_I4:
  case LSDA_I8:
  case LSDA_INT:
  case LSDA_SHORT:
  case LSDA_LONG:
  case LSDA_INTEGER:
    return LSDA_INT;
  case LSDA_U1:
  case LSDA_U2:
  case LSDA_U4:
  case LSDA_U8:
  case LSDA_UINT:
  case LSDA_USHORT:
  case LSDA_ULONG:
    return LSDA_UINT;
  case LSDA_R4:
  case LSDA_R8:
  case LSDA_FLOAT:
  case LSDA_DOUBLE:
  case LSDA_REAL:
  case LSDA_DP:
    return LSDA_FLOAT;
  }
  return 0;
}

int lsda_util_id2size(int type_id)
{
  switch(type_id)
  {
/* most common ones first */
  case LSDA_I4: return 4;
  case LSDA_R4: return 4;
  case LSDA_I8: return 8;
  case LSDA_R8: return 8;
  case LSDA_I1: return 1;
  case LSDA_I2: return 2;
  case LSDA_U4: return 4;
  case LSDA_U8: return 8;
  case LSDA_U1: return 1;
  case LSDA_U2: return 2;
  case LSDA_LINK: return 1;
/* I don't think these ever show up in a file in practice */
  case LSDA_SHORT: return sizeof(short);
  case LSDA_INT: return sizeof(int);
  case LSDA_INTEGER: return sizeof(FortranInteger);
  case LSDA_LONG: return sizeof(long);
  case LSDA_USHORT: return sizeof(unsigned short);
  case LSDA_UINT: return sizeof(unsigned int);
  case LSDA_ULONG: return sizeof(unsigned long);
  case LSDA_FLOAT: return sizeof(float);
  case LSDA_REAL: return sizeof(FortranReal);
  case LSDA_DOUBLE: return sizeof(double);
  case LSDA_DP: return sizeof(FortranDouble);
  }
  return 0;
}

int lsda_util_db2sg(int type_id)
{
  switch(type_id)
  {
  case LSDA_DOUBLE: return LSDA_FLOAT;
  case LSDA_DP: return LSDA_REAL;
  case LSDA_R8: return LSDA_R4;
  case LSDA_U8: return LSDA_U4;
  case LSDA_ULONG: return LSDA_UINT;
  case LSDA_I8: return LSDA_I4;
  case LSDA_LONG: return LSDA_INT;  
  }
  return type_id;
}
extern void free_all_tables(void);
extern void free_all_types(void);

#ifndef NO_FORTRAN

extern void free_all_fdirs(void);

#endif
void free_all_lsda(void)
{
  int i;
/*
  First close all open files
*/
  for(i=0; i<num_daf; i++) {
    if(!da_store[i].free) lsda_close(i);
  }
/*
  Now free everything
*/
  if(da_store) free(da_store);
  free_all_tables();
  free_all_types();
#ifndef NO_FORTRAN
  free_all_fdirs();
#endif
  da_store=NULL;
  num_daf = 0;
  _errno = ERR_NONE;
  report_level = 0;
}


// qd additions
char**
lsda_get_children_names(int handle, char* name, int follow, Length* length)
{
  length = 0;
  LSDAFile* daf;
  LSDATable* t;
  LSDATable* current_table;

  // get file
  if (handle < 0 || handle >= num_daf) {
    _errno = ERR_NOFILE;
    //*type_id = -1;
    *length = 0;
    if (report_level > 0)
      fprintf(stderr, "lsda_query: invalid handle %d", handle);
    return 0;
  }
  daf = da_store + handle;

  // find symbol
  t = daf->FindVar(daf, name, 0, follow);
  // check for error
  if (t) {
    if (t->type == 0) {
      // yay im a dir
      *length = BT_numentries(t->children);
    } else {
      // do something coz im a var, not a dir
      return 0; // return empty
    }
  } else {
    // do someting because i failed
    fprintf(stderr, "lsda_get_children_names: path %s does not exist.", name);
    return 0;
  }

  // no children
  if((*length) < 1)
    return 0;

  int count = 0;
  LSDATable* child = 0;
  char** ret = (char**) malloc((*length)*sizeof(char*));

  /*
  int l = (int) strlen(t->name);
  strcpy(q, t->name);
  if (!t->type) {
    // let string finish correctly
    if (t->name[l - 1] != '/') {
      q[l++] = '/';
      q[l] = '\0';
    }
  }
  */

  int iChild = 0;
  while(1) {
    child = (LSDATable*) BT_enumerate(t->children, &count);
    if (!child)
      break;

    char* entry = (char*) malloc(strlen(child->name)* sizeof(char));
    strcpy(entry, child->name);
    ret[iChild++] = entry;
  }

  return ret;
}