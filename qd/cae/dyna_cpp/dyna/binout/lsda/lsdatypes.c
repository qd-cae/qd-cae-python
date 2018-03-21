/*
  Copyright (C) 2002
  by Livermore Software Technology Corp. (LSTC)
  All rights reserved
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined WIN32 || defined WIN64
#include <windows.h>
#endif
#include "lsda.h"
#include "lsda_internal.h"

typedef struct _tf {
  void *p;
  struct _tf *next;
}       TO_FREE;
static TO_FREE          *to_free = NULL;
static LSDAType		*free_types=NULL;

/* Random, cyclic permutation of the first 256 integers and its inverse */
unsigned nums[256] = {
	103,126,26,40,211,23,196,244,138,172,213,203,192,183,130,149,
	80,14,69,181,75,111,147,233,219,125,254,232,123,44,28,35,
	133,61,25,30,158,93,156,22,139,205,176,85,204,66,99,39,
	246,88,13,159,188,101,226,110,92,157,209,175,24,220,161,98,
	0,33,6,250,91,57,124,100,71,253,236,121,215,52,152,73,
	21,255,200,144,248,162,145,238,163,140,46,243,42,5,212,119,
	210,137,128,51,189,63,50,198,122,115,83,150,134,182,3,34,
	62,206,129,81,65,216,90,108,218,36,166,197,194,245,151,164,
	165,113,116,47,173,7,43,112,38,249,10,86,12,118,76,56,
	87,114,221,184,117,177,143,234,224,15,102,180,82,174,9,201,
	120,225,4,37,19,89,55,191,153,8,58,242,16,222,104,60,
	59,235,155,239,17,78,193,49,109,195,171,168,95,31,97,214,
	131,199,84,217,132,20,105,237,29,154,11,208,223,186,167,96,
	72,230,53,54,178,68,146,45,190,160,106,2,135,136,48,64,
	77,252,240,70,79,202,148,141,231,94,107,241,41,18,207,247,
	228,179,251,142,185,127,32,229,27,187,170,227,1,169,67,74
};
unsigned inums[256] = {
	64,252,219,110,162,93,66,133,169,158,138,202,140,50,17,153,
	172,180,237,164,197,80,39,5,60,34,2,248,30,200,35,189,
	246,65,111,31,121,163,136,47,3,236,92,134,29,215,90,131,
	222,183,102,99,77,210,211,166,143,69,170,176,175,33,112,101,
	223,116,45,254,213,18,227,72,208,79,255,20,142,224,181,228,
	16,115,156,106,194,43,139,144,49,165,118,68,56,37,233,188,
	207,190,63,46,71,53,154,0,174,198,218,234,119,184,55,21,
	135,129,145,105,130,148,141,95,160,75,104,28,70,25,1,245,
	98,114,14,192,196,32,108,220,221,97,8,40,89,231,243,150,
	83,86,214,22,230,15,107,126,78,168,201,178,38,57,36,51,
	217,62,85,88,127,128,122,206,187,253,250,186,9,132,157,59,
	42,149,212,241,155,19,109,13,147,244,205,249,52,100,216,167,
	12,182,124,185,6,123,103,193,82,159,229,11,44,41,113,238,
	203,58,96,4,94,10,191,76,117,195,120,24,61,146,173,204,
	152,161,54,251,240,247,209,232,27,23,151,177,74,199,87,179,
	226,235,171,91,7,125,48,239,84,137,67,242,225,73,26,81
};

/* Scramble the bytes */
#define G1(i)	(int)((unsigned)(i) & 0xff)
#define G2(i)	(int)(((unsigned)(i) >> 8) & 0xff)
#define G3(i)	(int)(((unsigned)(i) >> 16) & 0xff)
#define G4(i)	(int)(((unsigned)(i) >> 24) & 0xff)
#define SCRAMBLE(i)	(nums[G1(i)] | \
			(nums[nums[G2(i)]] << 8) | \
			(nums[nums[nums[G3(i)]]] << 16) | \
			(nums[nums[nums[nums[G4(i)]]]] << 24))
#define UNSCRAMBLE(i)	(inums[G1(i)] | \
			(inums[inums[G2(i)]] << 8) | \
			(inums[inums[inums[G3(i)]]] << 16) | \
			(inums[inums[inums[inums[G4(i)]]]] << 24))

static LSDAType *
NewLSDAType(void)
{
 int	i;
 LSDAType	*type;
 TO_FREE	*tf;

 if(!free_types){
  free_types = (LSDAType *)malloc(sizeof(LSDAType) * LSDATYPE_GRAIN);
  if(!free_types){
   fprintf(stderr,"NewLSDAType: malloc failed\n");
   exit(1);
  }
  tf = (TO_FREE *)malloc(sizeof(TO_FREE));
  if(!tf){
   fprintf(stderr,"NewLSDAType: TO_FREE malloc failed\n");
   exit(1);
  }
  tf->p = free_types;
  tf->next = to_free;
  to_free=tf;
  for(i = 0; i < LSDATYPE_GRAIN-1; i++){
   free_types[i].right = &free_types[i+1];
  }
  free_types[LSDATYPE_GRAIN-1].right = NULL;
 }
 type = free_types;
 free_types = free_types->right;
 memset((char *)type,'\0',sizeof(LSDAType));
 type->alias = type;
 return(type);
}

static void
FreeLSDAType(LSDAType *type)
{
 if(!type) return;
 type->right = free_types;
 free_types = type;
}

static void
LSDAOrderTypeByID(LSDAFile *file,LSDAType *type)
{
 LSDAType		*tree;
 unsigned	id;
 unsigned	tid;

 tree = file->types;
 id = SCRAMBLE(type->id);
 while(1){
  tid = SCRAMBLE(tree->id);
  if(id < tid){
   if(!tree->idleft){
    tree->idleft = type;
    return;
   }
   tree = tree->idleft;
  }
  else if(id > tid){
   if(!tree->idright){
    tree->idright = type;
    return;
   }
   tree = tree->idright;
  }
  else{
   fprintf(stderr,"INTERNAL ERROR:LSDAOrderTypeByID found duplicate symbol\n");
   exit(1);
  }
 }
}

static LSDAType *
LSDAFindTypeByID(LSDAFile *file,int id)
{
 unsigned	newid,tid;
 LSDAType		*tree;

 newid = SCRAMBLE(id);
 tree = file->types;
 while(tree){
  tid = SCRAMBLE(tree->id);
  if(newid < tid)
   tree = tree->idleft;
  else if(newid > tid)
   tree = tree->idright;
  else
   return(tree->alias);
 }
 return(NULL);
}

static LSDAType *
_LSDAFindType(LSDAFile *file,char *name,int create)
{
 LSDAType		*type;
 int		ncmp;

 if(!file->types){
  type = file->types = NewLSDAType();
  strncpy(type->name,name,MAXNAME-1);
  type->name[MAXNAME-1]=0;
  type->id = ++file->ntypes;
  return(type);
 }
 type = file->types;

again:
 ncmp = strcmp(type->name,name);
 if(!ncmp){
  return(type->alias);
 }
 else if(ncmp < 0){
  if(!type->left){
   if(!create) return(NULL);
   type = type->left = NewLSDAType();
   strncpy(type->name,name,MAXNAME-1);
   type->name[MAXNAME-1]=0;
   type->id = ++file->ntypes;
   LSDAOrderTypeByID(file,type);
   return(type);
  }
  type = type->left;
  goto again;
 }
 else if(ncmp > 0){
  if(!type->right){
   if(!create) return(NULL);
   type = type->right = NewLSDAType();
   strncpy(type->name,name,MAXNAME-1);
   type->name[MAXNAME-1]=0;
   type->id = ++file->ntypes;
   LSDAOrderTypeByID(file,type);
   return(type);
  }
  type = type->right;
  goto again;
 }
 return NULL;  /* can never get here, but SGI complains....*/
}

static LSDAType *
LSDACreateType(LSDAFile *file,char *name)
{
 return(_LSDAFindType(file,name,TRUE));
}

static LSDAType *
LSDAFindType(LSDAFile *file,char *name)
{
 return(_LSDAFindType(file,name,FALSE));
}

static void
_LSDAFreeTypes(LSDAType *type)
{
 if(type->left) _LSDAFreeTypes(type->left);
 if(type->right) _LSDAFreeTypes(type->right);
 FreeLSDAType(type);
}
static void
LSDAFreeTypes(LSDAFile *file)
{
 if(file->types) _LSDAFreeTypes(file->types);
 file->types = NULL;
}

void
InitTypePointers(LSDAFile *file)
{
 file->CreateType	= LSDACreateType;
 file->FindType		= LSDAFindType;
 file->FindTypeByID	= LSDAFindTypeByID;
 file->FreeTypes	= LSDAFreeTypes;
}
/*
  This should ONLY be called after all the files are closed, and so
  we don't have to worry about following and freeing all the links
*/
void free_all_types(void)
{
TO_FREE *tf, *next;

for(tf=to_free; tf; tf=next) {
  next = tf->next;
  free(tf->p);
  free(tf);
}
to_free = NULL;
free_types=NULL;
}
