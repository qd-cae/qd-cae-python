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

typedef struct _tf
{
  void* p;
  struct _tf* next;
} TO_FREE;
static TO_FREE* to_free = NULL;
static LSDATable* free_table = NULL;

static LSDATable*
NewLSDATable(void)
{
  LSDATable* ret;
  TO_FREE* tf;

  if (!free_table) {
    int i;

    free_table = (LSDATable*)malloc(LSDATABLE_GRAIN * sizeof(LSDATable));
    if (!free_table) {
      fprintf(stderr, "NewLSDATable: malloc failed\n");
      exit(1);
    }
    tf = (TO_FREE*)malloc(sizeof(TO_FREE));
    if (!tf) {
      fprintf(stderr, "NewLSDATable: TO_FREE malloc failed\n");
      exit(1);
    }
    tf->p = free_table;
    tf->next = to_free;
    to_free = tf;
    for (i = 0; i < LSDATABLE_GRAIN - 1; i++) {
      free_table[i].next = &free_table[i + 1];
    }
    free_table[LSDATABLE_GRAIN - 1].next = NULL;
  }
  ret = free_table;
  free_table = ret->next;
  memset((char*)ret, '\0', sizeof(LSDATable));
  return (ret);
}

static void
FreeLSDATable(LSDATable* old)
{
  if (!old)
    return;
  old->next = free_table;
  if (old->children)
    BT_free(old->children);
  old->children = NULL;
  free_table = old;
}

static char**
LSDASplitPath(char* s, char** c)
/*
 * Internal utility used to split
 * a path into its components.
 */
{
  int nc = 0;
  char* q;
  static char* root = "/";

  q = &s[0];
  if (*q == '/') {
    c[nc++] = root;
    q++;
  }
  if (*q)
    c[nc++] = q;
  while (*q) {
    if (*q != '/') {
      if (q - c[nc - 1] >= MAXNAME - 1)
        *(q++) = '\0';
      else
        q++;
    } else {
      *(q++) = '\0';
      if (*q)
        c[nc++] = q;
    }
  }
  c[nc] = NULL;
  return (&c[0]);
}

static int
my_compfunc(void* p1, void* p2)
{
  LSDATable* d1 = (LSDATable*)p1;
  LSDATable* d2 = (LSDATable*)p2;
  return strcmp(d1->name, d2->name);
}

static LSDATable*
LSDAFind(LSDAFile* file, char* name, void* type, int create, int follow)
/*
 * Searches for a named directory or variable
 * from a path name which is relative to the
 * current directory.  If "create" is set,
 * and if the directory or path does not exist,
 * then the specified type is created provided
 * no component of the path which should be a
 * directory turns out to be a variable.
 */
{
  char **args, *c[64];
  char s[MAXPATH];
  LSDATable* start;
  LSDATable *t, *child, dummy;
  extern LSDATable* LSDAresolve_link(LSDAFile*, LSDATable*);

  /* Keep from crashing */
  if (!name || !*name) {
    /*
      sprintf(file->err_string,"LSDAFind called for NULL or empty name");
    */
    return (NULL);
  }

  /* Split the path into components and treat "/" carefully */
  strcpy(s, name);
  args = LSDASplitPath(s, c);
  start = file->cwd;
  if (!strcmp(args[0], "/")) {
    start = file->top;
    args++;
  }

  while (1) {
  again:
    if (!*args) {
      if (follow && start && start->type && start->type->id == LSDA_LINK)
        start = LSDAresolve_link(file, start);
      return start;
    }
    if (!strcmp(*args, ".")) {
      args++;
      goto again;
    }
    if (!strcmp(*args, "..")) {
      if (start->parent)
        start = start->parent;
      args++;
      goto again;
    }

    if (follow && start->type &&
        start->type->id == LSDA_LINK) { /* resolve symbolic links */
      start = LSDAresolve_link(file, start);
    }
    /* It is an error to encounter other than a directory */
    if (start->type) {
      /*
         sprintf(file->err_string,
              "LSDAFind: component \"%s\" of \"%s\" is not a directory",
              args[0],name);
      */
      return (NULL);
    }
    if (!strcmp(*args, "."))
      return (start); /* ??? */

    /* Add the directory if there are no subdirectories */
    if (!start->children) {
      if (!create) {
        /*
            sprintf(file->err_string,
                "LSDA_ERROR: LSDAFind: path not found: %s",name);
        */
        return (NULL);
      }
      start->children = BT_new(my_compfunc);
      child = NewLSDATable();
      strcpy(child->name, args[0]);
      child->parent = start;
      if (!args[1])
        child->type = (LSDAType*)type;
      BT_store(start->children, child);
      start = child;
      args++;
      if (child->type)
        return child; /* return entry just created */
      goto again;
    }

    /* Search the subdirectory for the specified entry */
    strcpy(dummy.name, args[0]);
    t = (LSDATable*)BT_lookup(start->children, &dummy, 0);
    if (t) {
      start = t;
      args++;
      goto again;
    }

    /* Add the directory at the end if no match was found */
    if (!create) {
      /*
         sprintf(file->err_string,
              "LSDAFind: path not found: %s",name);
      */
      return (NULL);
    }
    child = NewLSDATable();
    strcpy(child->name, args[0]);
    child->parent = start;
    if (!args[1])
      child->type = (LSDAType*)type;
    BT_store(start->children, child);
    start = child;
    args++;
    if (child->type)
      return child; /* return entry just created */
  }
}

LSDATable*
LSDAFindVar(LSDAFile* file, char* name, int create, int follow)
{
  return LSDAFind(file, name, NULL, create, follow);
}

static LSDATable*
LSDAChangeDir(LSDAFile* file, char* name, int create)
/*
 * Changes to a given directory specified by
 * the pathname "name".  If the "create" flag
 * is set, then all components of the path
 * which do not exist are automatically created.
 *
 * ERRORS: If a component of the path is not a
 * directory, then this routine returns NULL.
 */
{
  LSDATable* dt;

  dt = LSDAFind(file, name, NULL, create, 1);
  if (!dt)
    return NULL; /* not found */
  if (dt->type)
    return NULL; /* attempt to cd into a variable! */
  file->cwd = dt;
  return (dt);
}

static LSDATable*
LSDACreateVar(LSDAFile* file, LSDAType* type, char* name)
/*
 * Creates a variable specified by the pathname "name".
 * The type member must be non-NULL, or this routine
 * will create a directory instead.  Any components of
 * the path which do not exist are created automatically.
 *
 * ERRORS: If a non-final component of the path is not
 * a directory, or if the final component exists and
 * is not of the specified type, then this routine
 * returns a NULL pointer.
 */
{
  LSDATable* dt;

  dt = LSDAFind(file, name, type, TRUE, 0);
  if (!dt)
    return (dt);
  if (dt->type != type) {
    /*
      sprintf(file->err_string,
            "LSDACreateVar: incompatible type for \"%s\"",name);
    */
    return (NULL);
  }
  return (dt);
}
static LSDATable*
LSDACreateVar2(LSDAFile* file, char* type, char* name)
{
  LSDAType* new_type;

  new_type = file->FindType(file, type);
  if (!new_type) {
    /*
      sprintf(file->err_string,
            "LSDACreateVar: unknown type of variable \"%s\"",type);
    */
    return (NULL);
  }
  return LSDACreateVar(file, new_type, name);
}

static LSDATable*
LSDACreateDir(LSDAFile* file, char* name)
/*
 * Creates a directory specified by the pathname "name".
 * Any component of the path which does not exist is
 * automatically created.
 *
 * ERRORS: If any component of the path already exists
 * but is not a directory, then this routine returns NULL.
 */
{
  LSDATable* dt;

  dt = LSDAFind(file, name, NULL, TRUE, 1);
  if (!dt)
    return (dt);
  if (dt->type) {
    /*
      sprintf(file->err_string,
            "LSDACreateDirectory: \"%s\" already exists as a variable\n",name);
    */
    return (NULL);
  }
  return (dt);
}

static char*
LSDAGetPath(LSDAFile* file, LSDATable* t)
/*
 * Constructs the fully-qualified path name
 * of the current working directory and returns
 * a pointer to static memory containing the
 * full path name.
 */
{
  LSDATable* dt;
  char* q;
  static char path[MAXPATH];
  int l;

  if (!t) {
    /*
      sprintf(file->err_string,"LSDAGetPath: NULL argument");
    */
    return (NULL);
  }
  if (!t->parent) { /* root directory */
    path[0] = '/';
    path[1] = '\0';
    return path;
  }
  q = &path[MAXPATH - 1];
  *(q--) = '\0';
  for (dt = t; dt->parent; dt = dt->parent) {
    q -= l = (int)strlen(&dt->name[0]);
    if (q < &path[0]) {
      /*
         sprintf(file->err_string,"LSDAGetPath: path too long ...%s",q);
      */
      return (NULL);
    }
    strncpy(q, &dt->name[0], l);
    if (*q != '/') {
      if (q == &path[0]) {
        /*
            sprintf(file->err_string,"LSDAGetPath: path too long ...%s",q);
        */
        return (NULL);
      }
      *(--q) = '/';
    }
  }
  return (q);
}

static char*
LSDAGetCWD(LSDAFile* file)
/*
 * Returns the fully qualified pathname of
 * the current working directory.
 *
 * ERRORS: Returns NULL if the current path
 * is not set, or if the fully qualified pathname
 * exceeds MAXPATH-1 characters.
 */
{
  return (LSDAGetPath(file, file->cwd));
}

static char _tmp[MAXPATH];
static void
_LSDAPrintTable(LSDATable* t, char* q)
{
  int l;
  int cont;
  LSDATable* child;

  l = (int)strlen(t->name);
  strcpy(q, t->name);
  if (!t->type) {
    if (t->name[l - 1] != '/') {
      q[l++] = '/';
      q[l] = '\0';
    }
  }
  if (t->children) {
    for (cont = 0;;) {
      child = (LSDATable*)BT_enumerate(t->children, &cont);
      if (!child)
        break;
      _LSDAPrintTable(child, q + l);
    }
  }
  q[l] = '\0';
  printf("%s\n", _tmp);
}

static void
LSDAPrintTable(LSDAFile* file, LSDATable* t)
/*
 * Prints all pathnames of directories
 * and variables starting at the node "t".
 */
{
  _tmp[0] = '\0';
  _LSDAPrintTable(t, &_tmp[0]);
}

static void
_LSDATableFree(LSDATable* t)
{
  LSDATable* child;
  int cont;

  if (t->children) {
    for (cont = 0;;) {
      child = (LSDATable*)BT_enumerate(t->children, &cont);
      if (!child)
        break;
      _LSDATableFree(child);
    }
  }
  FreeLSDATable(t);
}
static void
LSDATableFree(LSDAFile* file, LSDATable* t)
/*
 * Frees any node by unlinking it from its parent
 * directory (if there is one).  And if the node
 * is a directory, then all children of that
 * directory are recursively freed.
 */
{
  if (!t)
    return;
  if (t->parent && t->parent->children)
    BT_delete(t->parent->children, t);
  _LSDATableFree(t);
}

void
InitLSDAFile(LSDAFile* file)
/*
 * initializes the structure and its
 * member functions.
 */
{
  extern void InitTypePointers(LSDAFile*);

  memset((char*)file, '\0', sizeof(LSDAFile));
  file->top = NewLSDATable();
  strcpy(file->top->name, "/");
  file->cwd = file->top;
  file->CreateDir = LSDACreateDir;
  file->ChangeDir = LSDAChangeDir;
  file->CreateVar = LSDACreateVar;
  file->CreateVar2 = LSDACreateVar2;
  file->FreeTable = LSDATableFree;
  file->GetCWD = LSDAGetCWD;
  file->GetPath = LSDAGetPath;
  file->PrintTable = LSDAPrintTable;
  file->ntypes = 0;
  file->FindVar = LSDAFindVar;
  InitTypePointers(file);
}

LSDAFile*
NewLSDAFile(void)
/*
 * Creates a new LSDAFile structure and
 * initializes the structure and its
 * member functions.
 */
{
  LSDAFile* file;
  extern void InitTypePointers(LSDAFile*);

  file = (LSDAFile*)malloc(sizeof(LSDAFile));
  if (!file) {
    fprintf(stderr, "NewLSDAFile: malloc failed\n");
    exit(1);
  }
  memset((char*)file, '\0', sizeof(LSDAFile));
  file->top = NewLSDATable();
  strcpy(file->top->name, "/");
  file->cwd = file->top;
  file->CreateDir = LSDACreateDir;
  file->ChangeDir = LSDAChangeDir;
  file->CreateVar = LSDACreateVar;
  file->CreateVar2 = LSDACreateVar2;
  file->FreeTable = LSDATableFree;
  file->GetCWD = LSDAGetCWD;
  file->GetPath = LSDAGetPath;
  file->PrintTable = LSDAPrintTable;
  file->ntypes = 0;
  file->FindVar = LSDAFindVar;
  InitTypePointers(file);
  return (file);
}

void
FreeLSDAFile(LSDAFile* file)
{
  file->FreeTable(file, file->top);
  file->FreeTypes(file);
  free((char*)file);
}

#ifdef NEED_MAIN
main(int argc, char* argv[])
{
  LSDAFile* file;
  LSDAType* type;
  LSDATable* dt;
  int i;
  static char* vars[] = { "/bb", "/a/b/c/r/s", NULL };
  static char* names[] = { "/a",          "/a/a",     "/a/b",
                           "/a/c",        "/a/d",     "/b/b1/b2/b3/b4",
                           "/b/b2/b3/b4", "/b/b2/b1", "/b/b1/b3",
                           "/b/b1/../b2", "/../c",    "/../c/d",
                           "/a/b/c/r/s",  "/c/d",     "e1",
                           "../e2",       "../e3",    "../e2",
                           NULL };
  static LSDAType any;

  file = NewLSDAFile();
  type = file->CreateType(file, "REAL");
  type->length_on_disk = type->length = 4;
  type = file->CreateType(file, "real");
  type->length_on_disk = type->length = 4;
  type = file->CreateType(file, "INTEGER");
  type->length_on_disk = type->length = 4;
  type = file->CreateType(file, "integer");
  type->length_on_disk = type->length = 4;
  type = file->CreateType(file, "REAL*8");
  type->length_on_disk = type->length = 8;
  type = file->CreateType(file, "real*8");
  type->length_on_disk = type->length = 8;
  type = file->CreateType(file, "DOUBLE PRECISION");
  type->length_on_disk = type->length = 8;
  type = file->CreateType(file, "double precision");
  type->length_on_disk = type->length = 8;
  for (i = 0; vars[i]; i++) {
    dt = file->CreateVar2(file, "REAL", vars[i]);
    if (!dt)
      printf("%s\n", file->err_string);
    printf("PATH:%s\n", file->GetPath(file, dt));
  }
  for (i = 0; names[i]; i++) {
    dt = file->ChangeDir(file, names[i], TRUE);
    if (!dt)
      printf("%s\n", file->err_string);
    printf("PATH:%s", file->GetPath(file, dt));
    if (dt) {
      if (!dt->type)
        printf(" - directory\n");
      else
        printf(" - variable\n");
    }
  }
  file->PrintTable(file, file->top);
}
#endif
/*
  This should ONLY be called after all the files are closed, and so
  we don't have to worry about following and freeing all the links
*/
void
free_all_tables(void)
{
  TO_FREE *tf, *next;

  for (tf = to_free; tf; tf = next) {
    next = tf->next;
    free(tf->p);
    free(tf);
  }
  to_free = NULL;
  free_table = NULL;
}
