#ifndef LSDA_INTERNAL_H
#define LSDA_INTERNAL_H

#include "define.h"

#define _HAVE_R4
typedef float TYPE_R4;
#define _HAVE_R8
typedef double TYPE_R8;
#define FP_FORMAT 0 /* 0 = IEEE, others someday....???? */

typedef void (*_CF)(void* vp1, void* pv2, int count);

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#define MAXPATH 2048
#define MAXNAME 64
#define LSDATABLE_GRAIN 200
#define LSDATYPE_GRAIN 50
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#include "btree.h"

#ifdef HAVE_AES
#include "aes.h"
#endif

typedef struct _IFile
{
  char* dirname;       /* directory where files live */
  char* filename;      /* actual file name */
  size_t stoffset;     /* location of last chunk of symbol table */
  int fp_format;       /* value of FP_FORMAT when file written */
  int bigendian;       /* value of BIGENDIAN when file written */
  int FileLengthSize;  /* sizeof(Length) when file written */
  int FileOffsetSize;  /* sizeof(Offset) when file written */
  int FileCommandSize; /* sizeof(Command) when file written */
  int FileTypeIDSize;  /* sizeof(TypeID) when file written */
  int ateof;           /* true if last read/write left this file at EOF */
  _CF ConvertLength;   /* conversion functions for these */
  _CF ConvertOffset;
  _CF ConvertCommand;
  _CF ConvertTypeID;
} IFile; /* an individual file on disk */

typedef struct _LSDAType
{
  char name[MAXNAME];
  char* def;
  int length;
  int length_on_disk;
  LSDA_TypeID id; /* Sequential Id assigned at creation */
  void (*Convert)(char* input, char* output, int count);
  struct _LSDAType* left;    /* for the tree structure only */
  struct _LSDAType* right;   /* for the tree structure only */
  struct _LSDAType* idleft;  /* for sorting by id */
  struct _LSDAType* idright; /* for sorting by id */
  struct _LSDAType* alias;   /* for aliasing data types */
} LSDAType;

#define LSDAId(type) ((type)->id)
#define LSDASizeOf(type) ((type)->length)
#define LSDASizeOfOnDisk(type) ((type)->length_on_disk)

typedef struct _LSDATable
{
  char name[MAXNAME];
  LSDAType* type;
  IFile* ifile; /* which ifile var is in (reading only) */
  int dirty;    /* 1 iff symbol table entry needs to be written to disk */
  size_t offset;
  size_t length;
  struct _LSDATable* parent;
  struct _LSDATable* next; /* for memory allocation */
  void* children;          /* btree pointer */
} LSDATable;

#ifndef HAVE_AES
typedef int aes_ctx;
#endif

typedef struct _LSDAFile
{
  LSDATable* top;
  LSDATable* (*CreateDir)(struct _LSDAFile*, char* name);
  LSDATable* (*ChangeDir)(struct _LSDAFile*, char* name, int create);
  LSDATable* (*CreateVar)(struct _LSDAFile*, LSDAType* type, char* name);
  LSDATable* (*CreateVar2)(struct _LSDAFile*, char* type, char* name);
  LSDATable* (*FindVar)(struct _LSDAFile*, char* name, int create, int follow);
  void (*FreeTable)(struct _LSDAFile*, LSDATable*);
  char* (*GetPath)(struct _LSDAFile*, LSDATable*);
  void (*PrintTable)(struct _LSDAFile*, LSDATable*);
  LSDATable* cwd;
  char* (*GetCWD)(struct _LSDAFile*);
  LSDAType* types;
  int ntypes; /* number of types defined so far */
  LSDAType* (*CreateType)(struct _LSDAFile*, char* type);
  LSDAType* (*FindType)(struct _LSDAFile*, char* type);
  LSDAType* (*FindTypeByID)(struct _LSDAFile*, int id);
  void (*FreeTypes)(struct _LSDAFile*);
  LSDATable* var;            /* var currently being written/last written */
  FILE* fpw;                 /* the actual file pointer for writing */
  FILE* fpr;                 /* the actual file pointer for reading */
  IFile** ifile;             /* array of IFile pointers for the actual files */
  int num_list;              /* # of ifiles in list */
  char lastpath[MAXPATHLEN]; /* dir of last var written to file */
  int continued;             /* true if currently continuing a var */
  int free;                  /* true if this struct is currently unused */
  IFile* ifr;                /* current file opened for reading */
  IFile* ifw;                /* current file opened for writing */
  int openmode;           /* how file was opened (READ, WRITE, or READWRITE) */
  int pathchanged;        /* flag to indicate CWD needs updating */
  int stpendlen;          /* length of pending Symbol Table modifications */
  int encrypted;          /* flag for use of encryption */
  size_t maxsize;         /* maximum file size */
  unsigned char salt[16]; /* encryption salt */
  unsigned char pending[16]; /* data pending for encrypted output */
  int npend;                 /* # bytes of pending data */
  aes_ctx ctx[1];            /* encryption context */
} LSDAFile;
extern LSDAFile*
NewLSDAFile(void);
extern void
InitLSDAFile(LSDAFile*);
extern void
FreeLSDAFile(LSDAFile*);

#define BUFSIZE 4096

#define LSDA_NULL 1
#define LSDA_CD 2
#define LSDA_DATA 3
#define LSDA_VARIABLE 4
#define LSDA_BEGINSYMBOLTABLE 5
#define LSDA_ENDSYMBOLTABLE 6
#define LSDA_SYMBOLTABLEOFFSET 7

/*
  This constant determines the default maximum file size.  It should
  be smaller than 2^32 bytes, since on some machines an Offset may only
  be 4 bytes.  Plus we want to allow a large margin in case some huge
  things get written to the file...so I'm thinking 3GB
*/
#define DEF_MAX_SIZE 3221225472

/*
  Error handling
*/

#define ERR_NONE 0                /* no error */
#define ERR_MALLOC 1              /* malloc failed */
#define ERR_NOFILE 2              /* non-existent file */
#define ERR_FSEEK 3               /* fseek failed */
#define ERR_READ 4                /* read error on file */
#define ERR_WRITE 5               /* write error on file */
#define ERR_NOENDSYMBOLTABLE 6    /* append but end of symbol table not found */
#define ERR_OPENDIR 7             /* error opening directory for file */
#define ERR_OPENFILE 8            /* error opening file */
#define ERR_NOCONT 9              /* empty name to write when not continuing */
#define ERR_DATATYPE 10           /* write with unknown data type */
#define ERR_NOTYPEID 11           /* read unknown type id from file */
#define ERR_CD 12                 /* illegal cd attempt in file */
#define ERR_CLOSE 13              /* error on close ?? */
#define ERR_NOVAR 14              /* read on non-existant variable */
#define ERR_NOBEGINSYMBOLTABLE 15 /* file missing BEGINSYMBOLTABLE */
#define ERR_OPNDIR 16             /* open directory in file for querry */

#endif
