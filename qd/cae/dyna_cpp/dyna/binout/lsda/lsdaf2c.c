/*
  Copyright (C) 2002
  by Livermore Software Technology Corp. (LSTC)
  All rights reserved
*/
#define __BUILD_LSDAF2C__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined _WIN32 && !defined WIN64 && !defined MPPWIN
#include <dirent.h>
#include <unistd.h>
#define DIR_SEP '/'
#else
#include <direct.h>
#include <windows.h>
#define DIR_SEP '\\'
#define _errno win_errno
#endif
#include "lsda.h"
#include "lsda_internal.h"

#define Offset LSDA_Offset
#define Length LSDA_Length

#ifndef FortranI8
#define FortranI8 long long
#endif

#ifdef UNSCORE
#define LSDA_REOPEN_C lsda_reopen_c_
#define LSDA_FOPEN_C lsda_fopen_c_
#define LSDA_TRUNCATE_C lsda_truncate_c_
#define LSDA_OPEN_C lsda_open_c_
#define LSDA_OPEN_MANY_C lsda_open_many_c_
#define LSDA_OPEN_MANY_AES_C lsda_open_many_aes_c_
#define LSDA_TEST_C lsda_test_c_
#define LSDA_REOPEN_C_AES lsda_reopen_c_aes_
#define LSDA_FOPEN_C_AES lsda_fopen_c_aes_
#define LSDA_TRUNCATE_C_AES lsda_truncate_c_aes_
#define LSDA_OPEN_C_AES lsda_open_c_aes_
#define LSDACLOSE lsdaclose_
#define LSDA_FSIZE_C lsda_fsize_c_
#define LSDAFILENUM lsdafilenum_
#define LSDANEXTFILE lsdanextfile_
#define LSDASETMAXSIZE lsdasetmaxsize_
#define LSDA_CD_C lsda_cd_c_
#define LSDAFLUSH lsdaflush_
#define LSDASYNC lsdasync_
#define LSDA_WRITE_C lsda_write_c_
#define LSDA_REWRITE_C lsda_rewrite_c_
#define LSDA_READ_C lsda_read_c_
#define LSDA_READ_C64 lsda_read_c64_
#define LSDA_LREAD_C lsda_lread_c_
#define LSDANEXTOPEN lsdanextopen_
#define LSDASETRLEVEL lsdasetrlevel_
#define LSDA_GETNAME_C lsda_getname_c_
#define LSDA_GETBASENAME_C lsda_getbasename_c_
#define LSDA_GETHANDLE_C lsda_gethandle_c_
#define LSDA_QUERY_C lsda_query_c_
#define LSDA_QUERY_C64 lsda_query_c64_
#define LSDA_LQUERY_C lsda_lquery_c_
#define LSDA_QUERYVAR_C lsda_queryvar_c_
#define LSDA_OPENDIR_C lsda_opendir_c_
#define LSDACLOSEDIR lsdaclosedir_
#define LSDA_READDIR_C lsda_readdir_c_
#define LSDAFREEALL lsdafreeall_
#define LSDA_COPYDIR_C lsda_copydir_c_
#define LSDA_GETPWD_C lsda_getpwd_c_
#define LSDAGETMODE lsdagetmode_
#define LSDAID2SIZE lsdaid2size_
#else
#ifndef UPCASE
#define LSDA_REOPEN_C lsda_reopen_c
#define LSDA_FOPEN_C lsda_fopen_c
#define LSDA_TRUNCATE_C lsda_truncate_c
#define LSDA_OPEN_C lsda_open_c
#define LSDA_OPEN_MANY_C lsda_open_many_c
#define LSDA_OPEN_MANY_AES_C lsda_open_many_aes_c
#define LSDA_TEST_C lsda_test_c
#define LSDA_REOPEN_C_AES lsda_reopen_c_aes
#define LSDA_FOPEN_C_AES lsda_fopen_c_aes
#define LSDA_TRUNCATE_C_AES lsda_truncate_c_aes
#define LSDA_OPEN_C_AES lsda_open_c_aes
#define LSDACLOSE lsdaclose
#define LSDA_FSIZE_C lsda_fsize_c
#define LSDAFILENUM lsdafilenum
#define LSDANEXTFILE lsdanextfile
#define LSDASETMAXSIZE lsdasetmaxsize
#define LSDA_CD_C lsda_cd_c
#define LSDAFLUSH lsdaflush
#define LSDASYNC lsdasync
#define LSDA_WRITE_C lsda_write_c
#define LSDA_REWRITE_C lsda_rewrite_c
#define LSDA_READ_C lsda_read_c
#define LSDA_READ_C64 lsda_read_c64
#define LSDA_LREAD_C lsda_lread_c
#define LSDANEXTOPEN lsdanextopen
#define LSDASETRLEVEL lsdasetrlevel
#define LSDA_GETNAME_C lsda_getname_c
#define LSDA_GETBASENAME_C lsda_getbasename_c
#define LSDA_GETHANDLE_C lsda_gethandle_c
#define LSDA_QUERY_C lsda_query_c
#define LSDA_QUERY_C64 lsda_query_c64
#define LSDA_LQUERY_C lsda_lquery_c
#define LSDA_QUERYVAR_C lsda_queryvar_c
#define LSDA_OPENDIR_C lsda_opendir_c
#define LSDACLOSEDIR lsdaclosedir
#define LSDA_READDIR_C lsda_readdir_c
#define LSDAFREEALL lsdafreeall
#define LSDA_COPYDIR_C lsda_copydir_c
#define LSDA_GETPWD_C lsda_getpwd_c
#define LSDAGETMODE lsdagetmode
#define LSDAID2SIZE lsdaid2size
#endif
#endif

typedef struct _ldc
{
  int used;
  LSDADir* dir;
} DPTR;

extern int
lsda_fopen_aes(char* filen,
               int filenum,
               Offset offset,
               int mode,
               int want,
               char* key);
extern int
lsda_reopen_aes(char* filen, int filenum, Offset offset, int mode, char* key);
extern int
lsda_truncate_aes(char* filen, int filenum, Offset offset, char* key);
extern int
lsda_filenum(int handle);
extern int
lsda_nextopen(int handle);
extern void
lsda_setreportlevel(int level);
extern int
lsda_open_aes(char* filen, int mode, char* key);
extern int
lsda_copydir(int h1, char* dir1, int h2, char* dir2);

static DPTR* mine = NULL;
static int mine_size = 0;

FortranInteger
LSDA_REOPEN_C(char* filen,
              FortranInteger* filenum,
              FortranInteger* offset,
              FortranInteger* mode,
              FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_reopen_aes(
    filen, (int)*filenum, (Offset)*offset, (int)*mode, NULL);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_REOPEN_C_AES(char* filen,
                  FortranInteger* filenum,
                  FortranInteger* offset,
                  FortranInteger* mode,
                  unsigned char* key,
                  FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_reopen_aes(
    filen, (int)*filenum, (Offset)*offset, (int)*mode, (char*)key);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_FOPEN_C(char* filen,
             FortranInteger* filenum,
             FortranInteger* offset,
             FortranInteger* mode,
             FortranInteger* handle,
             FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_fopen_aes(
    filen, (int)*filenum, (Offset)*offset, (int)*mode, (int)*handle, NULL);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_FOPEN_C_AES(char* filen,
                 FortranInteger* filenum,
                 FortranInteger* offset,
                 FortranInteger* mode,
                 FortranInteger* handle,
                 unsigned char* key,
                 FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_fopen_aes(filen,
                                                         (int)*filenum,
                                                         (Offset)*offset,
                                                         (int)*mode,
                                                         (int)*handle,
                                                         (char*)key);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_TRUNCATE_C(char* filen,
                FortranInteger* filenum,
                FortranInteger* offset,
                FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_truncate_aes(
    filen, (int)*filenum, (Offset)*offset, (char*)NULL);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_TRUNCATE_C_AES(char* filen,
                    FortranInteger* filenum,
                    FortranInteger* offset,
                    unsigned char* key,
                    FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_truncate_aes(
    filen, (int)*filenum, (Offset)*offset, (char*)key);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_OPEN_C(char* filen, FortranInteger* mode, FortranInteger* ierr)
{
  FortranInteger retval =
    (FortranInteger)lsda_open_aes(filen, (int)*mode, NULL);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_OPEN_C_AES(char* filen,
                FortranInteger* mode,
                unsigned char* key,
                FortranInteger* ierr)
{
  FortranInteger retval =
    (FortranInteger)lsda_open_aes(filen, (int)*mode, (char*)key);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_OPEN_MANY_C(char* filen, FortranInteger* num, FortranInteger* ierr)
{
  FortranInteger retval;
  char** f = (char**)malloc(*num * sizeof(char*));
  int i;

  f[0] = filen;
  for (i = 0; i < *num - 1; i++)
    f[i + 1] = f[i] + strlen(f[i]) + 1;

  retval = (FortranInteger)lsda_open_many_aes(f, (int)*num, NULL);
  free(f);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_OPEN_MANY_AES_C(char* filen,
                     FortranInteger* num,
                     unsigned char* key,
                     FortranInteger* ierr)
{
  FortranInteger retval;
  char** f = (char**)malloc(*num * sizeof(char*));
  int i;

  f[0] = filen;
  for (i = 0; i < *num - 1; i++)
    f[i + 1] = f[i] + strlen(f[i]) + 1;

  retval = (FortranInteger)lsda_open_many_aes(f, (int)*num, (char*)key);
  free(f);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
FortranInteger
LSDA_TEST_C(char* filen, FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_test_aes(filen, NULL);
  if (retval == 0)
    *ierr = -1;
  else
    *ierr = LSDA_SUCCESS;
  return *ierr;
}
void
LSDACLOSE(FortranInteger* handle, FortranInteger* ierr)
{
  int rc = lsda_close((int)*handle);
  if (rc == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  *handle = -1;
}
FortranI8
LSDA_FSIZE_C(FortranInteger* handle, FortranInteger* ierr)
{
  FortranI8 retval = lsda_fsize((int)*handle);
  *ierr = (FortranInteger)lsda_errno;
  return retval;
}
FortranInteger
LSDAFILENUM(FortranInteger* handle, FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_filenum((int)*handle);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
void
LSDANEXTFILE(FortranInteger* handle, FortranInteger* ierr)
{
  int retval = lsda_nextfile((int)*handle);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
FortranInteger
LSDAGETMODE(FortranInteger* handle, FortranInteger* ierr)
{
  FortranInteger retval = lsda_getmode((int)*handle);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
void
LSDASETMAXSIZE(FortranInteger* handle,
               FortranInteger* size,
               FortranInteger* ierr)
{
  int retval = lsda_setmaxsize((int)*handle, (Offset)*size);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDA_CD_C(FortranInteger* handle, char* path, FortranInteger* ierr)
{
  int retval = lsda_cd((int)*handle, path);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDAFLUSH(FortranInteger* handle, FortranInteger* ierr)
{
  int retval = lsda_flush((int)*handle);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDASYNC(FortranInteger* handle, FortranInteger* ierr)
{
  int retval = lsda_sync((int)*handle);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDA_WRITE_C(FortranInteger* handle,
             FortranInteger* type_id,
             char* name,
             FortranInteger* length,
             void* data,
             FortranInteger* ierr)
{
  int retval;

  if (*length < 0) {
    *ierr = ERR_WRITE;
    return;
  }

  retval = lsda_write((int)*handle, (int)*type_id, name, (Length)*length, data);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDA_REWRITE_C(FortranInteger* handle,
               FortranInteger* type_id,
               char* name,
               FortranInteger* start,
               FortranInteger* length,
               void* data,
               FortranInteger* ierr)
{
  int retval;

  if (*length < 0) {
    *ierr = ERR_WRITE;
    return;
  }

  retval = lsda_rewrite(
    (int)*handle, (int)*type_id, name, (Length)*start, (Length)*length, data);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDA_READ_C(FortranInteger* handle,
            FortranInteger* type_id,
            char* name,
            FortranInteger* offset,
            FortranInteger* number,
            void* data,
            FortranInteger* ierr)
{
  int retval = lsda_read(
    (int)*handle, (int)*type_id, name, (Length)*offset, (Length)*number, data);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDA_READ_C64(FortranInteger* handle,
              FortranInteger* type_id,
              char* name,
              FortranI8* offset,
              FortranInteger* number,
              void* data,
              FortranInteger* ierr)
{
  int retval = lsda_read(
    (int)*handle, (int)*type_id, name, (Length)*offset, (Length)*number, data);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
void
LSDA_LREAD_C(FortranInteger* handle,
             FortranInteger* type_id,
             char* name,
             FortranInteger* offset,
             FortranInteger* number,
             void* data,
             FortranInteger* ierr)
{
  int retval = lsda_lread(
    (int)*handle, (int)*type_id, name, (Length)*offset, (Length)*number, data);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
}
FortranInteger
LSDANEXTOPEN(FortranInteger* handle, FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_nextopen((int)*handle);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
void
LSDASETRLEVEL(FortranInteger* level)
{
  lsda_setreportlevel((int)*level);
}
void
LSDA_GETNAME_C(FortranInteger* handle,
               char* name,
               FortranInteger* len,
               FortranInteger* ierr)
{
  char* cp = lsda_getname((int)*handle);
  *len = strlen(cp);
  strcpy(name, cp);
  *ierr = LSDA_SUCCESS;
}
void
LSDA_GETBASENAME_C(FortranInteger* handle,
                   char* name,
                   FortranInteger* len,
                   FortranInteger* ierr)
{
  char* cp = lsda_getbasename((int)*handle);
  *len = strlen(cp);
  strcpy(name, cp);
  *ierr = LSDA_SUCCESS;
}
FortranInteger
LSDA_GETHANDLE_C(char* filen, FortranInteger* ierr)
{
  FortranInteger retval = (FortranInteger)lsda_gethandle(filen);
  if (retval == -1)
    *ierr = (FortranInteger)lsda_errno;
  else
    *ierr = LSDA_SUCCESS;
  return retval;
}
void
LSDA_QUERY_C(FortranInteger* handle,
             char* name,
             FortranInteger* type,
             FortranInteger* len,
             FortranInteger* ierr)
{
  int ltype;
  Length llen;
  lsda_query((int)*handle, name, &ltype, &llen);
  *type = (FortranInteger)ltype;
  *len = (FortranInteger)llen;
  *ierr = LSDA_SUCCESS;
}
void
LSDA_LQUERY_C(FortranInteger* handle,
              char* name,
              FortranInteger* type,
              FortranInteger* len,
              FortranInteger* ierr)
{
  int ltype;
  Length llen;
  lsda_lquery((int)*handle, name, &ltype, &llen);
  *type = (FortranInteger)ltype;
  *len = (FortranInteger)llen;
  *ierr = LSDA_SUCCESS;
}
void
LSDA_QUERY_C64(FortranInteger* handle,
               char* name,
               FortranInteger* type,
               FortranI8* len,
               FortranInteger* ierr)
{
  int ltype;
  Length llen;
  lsda_lquery((int)*handle, name, &ltype, &llen);
  *type = (FortranInteger)ltype;
  *len = (FortranI8)llen;
  *ierr = LSDA_SUCCESS;
}
void
LSDA_QUERYVAR_C(FortranInteger* handle,
                char* name,
                FortranInteger* type,
                FortranInteger* len,
                FortranInteger* filenum,
                FortranInteger* ierr)
{
  int ltype, lfnum;
  Length llen;
  lsda_queryvar((int)*handle, name, &ltype, &llen, &lfnum);
  *type = (FortranInteger)ltype;
  *len = (FortranInteger)llen;
  *filenum = (FortranInteger)lfnum;
  *ierr = LSDA_SUCCESS;
}

int
LSDA_OPENDIR_C(FortranInteger* handle, char* name, FortranInteger* ierr)
{
  int i;

  for (i = 0; i < mine_size; i++)
    if (mine[i].used == 0)
      break;

  if (i == mine_size) {
    mine = (struct _ldc*)realloc(mine, (mine_size + 10) * sizeof(DPTR));
    for (; i < mine_size + 10; i++)
      mine[i].used = 0;
    i = mine_size;
    mine_size += 10;
  }
  mine[i].dir = lsda_opendir(*handle, name);
  if (mine[i].dir == NULL) {
    *ierr = (FortranInteger)lsda_errno;
    return 0;
  } else {
    mine[i].used = 1;
    *ierr = LSDA_SUCCESS;
    return i + 1;
  }
}
void
LSDACLOSEDIR(FortranInteger* index, FortranInteger* ierr)
{
  int i = *index - 1;

  if (i >= 0 && i < mine_size && mine[i].used) {
    lsda_closedir(mine[i].dir);
    mine[i].used = 0;
  }
  *ierr = LSDA_SUCCESS;
}
void
LSDA_READDIR_C(FortranInteger* dir,
               char* name,
               FortranInteger* namelen,
               FortranInteger* type_id,
               FortranInteger* len,
               FortranInteger* filenum,
               FortranInteger* ierr)
{
  int index = *dir - 1;
  if (index >= 0 && index < mine_size && mine[index].used) {
    int ltype, lfnum;
    Length llen;
    lsda_readdir(mine[index].dir, name, &ltype, &llen, &lfnum);
    *namelen = strlen(name);
    *type_id = (FortranInteger)ltype;
    *len = (FortranInteger)llen;
    *filenum = (FortranInteger)lfnum;
  } else {
    *namelen = 0;
    *type_id = -1;
    *len = -1;
    *filenum = -1;
  }
  *ierr = LSDA_SUCCESS;
}
void
LSDA_COPYDIR_C(FortranInteger* h1,
               char* dir1,
               FortranInteger* h2,
               char* dir2,
               FortranInteger* ierr)
{
  *ierr = lsda_copydir((int)*h1, dir1, (int)*h2, dir2);
}
void
LSDA_GETPWD_C(FortranInteger* handle,
              char* name,
              FortranInteger* len,
              FortranInteger* ierr)
{
  char* cp = lsda_getpwd((int)*handle);
  *len = strlen(cp);
  strcpy(name, cp);
  *ierr = LSDA_SUCCESS;
}
FortranInteger
LSDAID2SIZE(FortranInteger* type)
{
  return (FortranInteger)lsda_util_id2size((int)*type);
}

void
free_all_fdirs()
{
  if (mine_size > 0 && mine != NULL)
    free(mine);
}
void
LSDAFREEALL()
{
  free_all_lsda();
}
