/*
  Copyright (C) 2002
  by Livermore Software Technology Corp. (LSTC)
  All rights reserved
*/
#define __BUILD_BTREE__
#include <stdio.h>
#include <stdlib.h>
#if defined WIN32 || defined WIN64 || defined MPPWIN
#include <windows.h>
#include <direct.h>
#endif

#include "btree.h"

#ifdef VISIBLE
#define STATIC
#else
#define STATIC static
#endif

#define STACK_SIZE 64

typedef struct _ll {
#ifdef BTREE_DEBUG
 int height;
#endif
 int balance;
 void *data;
 struct _ll *left, *right;
} NODE;

typedef struct _stk {
  NODE **stack;
  int top;
} STACK;

typedef struct _btu {
 int n;           /* number of things in the table */
 STACK *left_off;  /* only used by enumeration routine */
 NODE **memory;
 int nmemory;
 NODE *unused;
 NODE *head;
 BT_CompFunc *compare;
} BTREE;

#ifdef BTREE_DEBUG

STATIC int bt_height(NODE *node)
{
/*
  Computes the height of a node in the tree, recursively
*/
  int left, right;
  if(node->left)
    left = bt_height(node->left);
  else
    left = 0;
  if(node->right)
    right = bt_height(node->right);
  else
    right = 0;
  node->height = (left > right ? left : right) + 1;
}

STATIC int bt_verify(BTREE *bt, NODE *node)
{
/*
  Checks the tree for errors:
    - the "balance" value at each node is correct
    - the elements in the tree are in the proper order
*/
  int left, right;
  if(node->left)
    left = node->left->height;
  else
    left = 0;
  if(node->right)
    right = node->right->height;
  else
    right = 0;
  if(node->balance != right - left) {
    printf("ERROR: imbalance at node with key %d\n",*(int *) node->data);
    printf("ERROR: right = %d, left = %d, balance=%d\n",right,left, node->balance);
    return -1;
  }
  if(node->left && (bt->compare(node->left->data,node->data) > 0)) {
    printf("ERROR: LEFT inequality at node with key %d\n",*(int *)node->data);
    return -1;
  }
  if(node->right && (bt->compare(node->right->data,node->data) < 0)) {
    printf("ERROR: RIGHT inequality at node with key %d\n",*(int *)node->data);
    return -1;
  }
  if(node->left && (bt_verify(bt,node->left) == -1)) {
    printf("  LEFT son of node with key %d\n",*(int *) node->data);
    return -1;
  }
  if(node->right && (bt_verify(bt,node->right) == -1)) {
    printf("  RIGHT son of node with key %d\n",*(int *) node->data);
    return -1;
  }
  return 1;
}
  
BT_verify(void *bti)
{
  BTREE *bt = (BTREE *) bti;
/*
  Compute height of every node
*/
  bt_height(bt->head);
  if(bt_verify(bt,bt->head) == -1) {
/* printf("TREE IS INVALID\n"); */
    return -1;
  } else {
/*    printf("Tree is valid\n");  */
    return 1;
  }
}
#endif

void *BT_new(BT_CompFunc *func)
{
/*
  Allocate a new binary tree structure
*/
  BTREE *bt;
/*
  Allocate btree
*/
  bt = (BTREE *) malloc(sizeof(BTREE));
  bt->n = 0;
  bt->left_off = NULL;
  bt->head = NULL;
  bt->unused = NULL;
  bt->memory = NULL;
  bt->nmemory = 0;
  bt->compare = func;
  return bt;
}

STATIC NODE *new_node(BTREE *bt,void *data)
{
/*
  Allocate a new node for the tree
*/
  NODE *node;
  int i,n;

  if(!bt->unused) {
/*
  Allocate a chunk of nodes.  Always allocate at least 5 but never more than 1000
*/
    n=bt->n/2;
    if(n < 5) n=5;
    if(n > 1000) n=1000;
    if(bt->nmemory++) {
      bt->memory = (NODE **) realloc(bt->memory,bt->nmemory*sizeof(NODE*));
    } else {
      bt->memory = (NODE **) malloc(sizeof(NODE*));
    }
    node = bt->memory[bt->nmemory-1] = (NODE *) malloc(n*sizeof(NODE));
    for(i=0; i<n-1; i++)
      node[i].right = node+i+1;
    node[n-1].right=NULL;
    bt->unused=node;
  }
  node=bt->unused;
  bt->unused=node->right;
  node->left = node->right = NULL;
  node->balance = 0;
  node->data = data;
  return node;
}

STATIC void rotate_left(NODE *node)
{
/*
  Helper function.  Rotates a portion of the tree, turning this:


                A
              B   C
                 D E

  Into this:


                C
              A   E
             B D

  Actually, A stays at the top, and C moves to where A is in the above diagram,
  but they swap their data pointers.  This way anything that points to the top of
  this subtree need not change.

  A = the input node "node"
*/
  void *data = node->data;
  NODE *child = node->right;
  node->right = child->right;
  child->right = child->left;
  child->left = node->left;
  node->left = child;
  node->data = child->data;
  child->data = data;
}

STATIC void rotate_right(NODE *node)
{
/*
  Just like the above, but rotates in the opposite direction
*/
  void *data = node->data;
  NODE *child = node->left;
  node->left = child->left;
  child->left = child->right;
  child->right = node->right;
  node->right = child;
  node->data = child->data;
  child->data = data;
}

STATIC void adjust(NODE **stack,int top)
{
/*
  Adjust the tree so that it remains in balance.  Called after the insertion
  of a node that might cause imbalance

  It progresses down the stack (up the tree) of nodes that are parents of
  the node that was added.

*/
int i;
NODE *node;

for(i=top-2; i>=0; i--) {
  if(stack[i]->left == stack[i+1]) {
    stack[i]->balance--;
    if(stack[i]->balance == 0) return;
    if(stack[i]->balance == -1) continue;
/*
  Ok, my left subtree is too long, and we have to adjust here.  There
  are two cases: my child's left subtree is longer, or his right subtree
  is longer.
*/
    if(stack[i+1]->balance > 0) {  /* right subtree longer */
      node = stack[i+1]->right;
      if(node->balance > 0) {
        stack[i+1]->balance = 0;
        stack[i]->balance = 0;
        node->balance = -1;
      } else if(node->balance < 0) {
        stack[i+1]->balance = +1;
        stack[i]->balance = 0;
        node->balance = 0;
      } else {   /* just added this node */
        stack[i+1]->balance = 0;
        stack[i]->balance = 0;
        node->balance = 0;
      }
      rotate_left(stack[i+1]);
      rotate_right(stack[i]);
      return;
    } else {                       /* left subtree longer */
      stack[i]->balance = stack[i+1]->balance = 0;
      rotate_right(stack[i]);
      return;
    }
  } else {
    stack[i]->balance++;
    if(stack[i]->balance == 0) return;
    if(stack[i]->balance == 1) continue;
/*
  Ok, my right subtree is too long, and we have to adjust here.  There
  are two cases: my child's left subtree is longer, or his right subtree
  is longer.
*/
    if(stack[i+1]->balance < 0) {  /* left subtree longer */
      node = stack[i+1]->left;
      if(node->balance < 0) {
        stack[i+1]->balance = 0;
        stack[i]->balance = 0;
        node->balance = +1;
      } else if(node->balance > 0) {
        stack[i+1]->balance = -1;
        stack[i]->balance = 0;
        node->balance = 0;
      } else {  /* must have just added this node....*/
        stack[i+1]->balance = 0;
        stack[i]->balance = 0;
        node->balance = 0;
      }
      rotate_right(stack[i+1]);
      rotate_left(stack[i]);
      return;
    } else {                       /* right subtree longer */
      stack[i]->balance = stack[i+1]->balance = 0;
      rotate_left(stack[i]);
      return;
    }
  }
}
}

void *BT_lookup(void *bti,void *data,int store)
{
/*
  Lookup an entry in the tree.  If store != 0, then store the
  entry if it is not found.  Otherwise return NULL if it is not
  found.  If found, the data pointer is returned.

  I suppose there is no way to tell the difference between an entry
  that is not found and one that is found but has NULL data pointer,
  but I'm not going to worry about that today....
*/
  BTREE *bt = (BTREE *) bti;
  NODE *node = bt->head;
  NODE *stack[STACK_SIZE];
  int comp, top = 0;
/*
  Special case -- first entry added to the tree
*/
  if(node == NULL) {
    if(store) {
      bt->head = new_node(bt,data);
      bt->n = 1;
    }
    return NULL;
  }
/*
  Go down tree looking for the data
*/
  while(1) {
    comp = bt->compare(data,node->data);
    if(comp == 0) return node->data;
    stack[top++] = node;
    if(comp < 0) {
      if(node->left) {
        node=node->left;
      } else {
        if(store) {
          node->left = new_node(bt,data);
          node->balance--;
          bt->n++;
          if(node->balance < 0) adjust(stack,top);
        }
        return NULL;
      }
    } else {
      if(node->right) {
        node=node->right;
      } else {
        if(store) {
          node->right = new_node(bt,data);
          node->balance++;
          bt->n++;
          if(node->balance > 0) adjust(stack,top);
        }
        return NULL;
      }
    }
  }
}

STATIC void shortened(NODE **stack,int top)
{
/*
  Like "adjust" above, but this is called when a node has been
  removed from the (bottom of the) tree, and we might have
  imbalance due to the shortening of a path
*/
NODE *n, *s, *gs;

while(top > 0) {
  n=stack[--top];
  switch(n->balance) {
    case -2:
      s = n->left;
      if(s->balance == -1) {    /* single rotation cases */
        n->balance = 0;
        s->balance = 0;
        rotate_right(n);
      } else if(s->balance == 0) {  /* height unchanged -- done */
        n->balance = 1;
        s->balance = -1;
        rotate_right(n);
        return;
      } else {             /* double rotation cases */
        gs = s->right;
        if(gs->balance == -1) {
          gs->balance = 0;
          s->balance = 1;
          n->balance = 0;
        } else if(gs->balance == 0) {
          gs->balance = 0;
          s->balance = 0;
          n->balance = 0;
        } else {
          gs->balance = -1;
          s->balance = 0;
          n->balance = 0;
        }
        rotate_left(s);
        rotate_right(n);
      }
      if(top > 0) {
        if(stack[top-1]->left == n)
          stack[top-1]->balance++;
        else
          stack[top-1]->balance--;
      }
      break;
    case  2:
      s = n->right;
      if(s->balance == +1) {    /* single rotation cases */
        n->balance = 0;
        s->balance = 0;
        rotate_left(n);
      } else if(s->balance == 0) {  /* height unchanged -- done */
        n->balance = -1;
        s->balance = +1;
        rotate_left(n);
        return;
      } else {             /* double rotation cases */
        gs = s->left;
        if(gs->balance == +1) {
          gs->balance = 0;
          s->balance = -1;
          n->balance = 0;
        } else if(gs->balance == 0) {
          gs->balance = 0;
          s->balance = 0;
          n->balance = 0;
        } else {
          gs->balance = +1;
          s->balance = 0;
          n->balance = 0;
        }
        rotate_right(s);
        rotate_left(n);
      }
      if(top > 0) {
        if(stack[top-1]->right == n)
          stack[top-1]->balance--;
        else
          stack[top-1]->balance++;
      }
      break;
    case 0:  /* balanced node, but changed subtree height so propigate up */
      if(top > 0) {
        if(stack[top-1]->left == n)
          stack[top-1]->balance++;
        else
          stack[top-1]->balance--;
      }
      break;
    default:  /* unbalanced this node, but didn't change subtree height */
      return;
      break;
  }
}
}

STATIC void leftreplace(BTREE *bt,NODE *top,NODE **stack, int *stop)
{
/*
  Replace indicated node with the largest element in it's left subtree
*/
  NODE *node=top->left;
  NODE *parent = top;
  NODE *todel;
  stack[(*stop)++] = top;
  stack[(*stop)++] = node;
  while(node->right) {
    parent = node;
    node = node->right;
    stack[(*stop)++] = node;
  }
  (*stop)--;

  top->data = node->data;

  if(node->left) {
    todel = node->left;
    *node = *node->left;
    todel->right = bt->unused;
    bt->unused = todel;
    if(parent->left == node) {
      parent->balance++;
      if(parent->balance != 1) shortened(stack,*stop);
    } else {
      parent->balance--;
      if(parent->balance != -1) shortened(stack,*stop);
    }
  } else {
    node->right = bt->unused;
    bt->unused = node;
    if(parent->left == node) {
      parent->left = NULL;
      parent->balance++;
      if(parent->balance != 1) shortened(stack,*stop);
    } else {
      parent->right = NULL;
      parent->balance--;
      if(parent->balance != -1) shortened(stack,*stop);
    }
  }
}
STATIC void rightreplace(BTREE *bt,NODE *top, NODE **stack, int *stop)
{
/*
  Replace indicated node with the smallest element in it's right subtree
*/
  NODE *node=top->right;
  NODE *parent = top;
  NODE *todel;
  stack[(*stop)++] = top;
  stack[(*stop)++] = node;
  while(node->left) {
    parent = node;
    node = node->left;
    stack[(*stop)++] = node;
  }
  (*stop)--;

  top->data = node->data;

  if(node->right) {
    todel = node->right;
    *node = *node->right;
    todel->right = bt->unused;
    bt->unused = todel;
    if(parent->right == node) {
      parent->balance--;
      if(parent->balance != -1) shortened(stack,*stop);
    } else {
      parent->balance--;
      if(parent->balance != +1) shortened(stack,*stop);
    }

  } else {
    node->right = bt->unused;
    bt->unused = node;
    if(parent->right == node) {
      parent->right = NULL;
      parent->balance--;
      if(parent->balance != -1) shortened(stack,*stop);
    } else {
      parent->left = NULL;
      parent->balance++;
      if(parent->balance != +1) shortened(stack,*stop);
    }
  }
}

void BT_delete(void *bti, void *data)
{
/*
  Go down the tree looking for this member.  If we find it,
  remove it.
*/
  BTREE *bt = (BTREE *) bti;
  NODE *node = bt->head;
  NODE *stack[STACK_SIZE];
  int comp,top=0;
  while(1) {
    comp = bt->compare(data,node->data);
    if(comp < 0) {
      if(node->left) {
        stack[top++] = node;
        node=node->left;
      } else {
        return;   /* not found */
      }
    } else if(comp > 0) {
      if(node->right) {
        stack[top++] = node;
        node=node->right;
      } else {
        return;   /* not found */
      }
    } else {
/*
  Found it.
*/
      if(node->left) {
        leftreplace(bt,node,stack,&top);
      } else if(node->right) {
        rightreplace(bt,node,stack,&top);
      } else {
        node->right = bt->unused;
        bt->unused = node;
        if(top) {
          if(stack[--top]->left == node) {
            stack[top]->left = NULL;
            stack[top]->balance++;
            if(stack[top]->balance != +1) shortened(stack,top+1);
          } else {
            stack[top]->right = NULL;
            stack[top]->balance--;
            if(stack[top]->balance != -1) shortened(stack,top+1);
          }
        } else {
          bt->head = NULL;
        }
      }
      bt->n--;
      return;
    }
  }
}

int BT_numentries(void *bti)
{
  BTREE *bt = (BTREE *) bti;
  return bt->n;
}

void *BT_enumerate(void *bti, int *cont)
{
/*
  Returns each entry of the tree in turn (in ascending sort order).  When
  we have no more entries to return, *cont is returned as 0.
*/
BTREE *bt = (BTREE *) bti;
NODE *node, *parent;
NODE *ret;

if(*cont == 0) {  /* initialize */
   if(bt->head == NULL) return NULL;
   *cont = 1;
  /*
    Initial call: find the first entry
  */
  if(!bt->left_off) {
    bt->left_off = (STACK *) malloc(sizeof(STACK));
    bt->left_off->stack = (NODE **) malloc(STACK_SIZE*sizeof(NODE *));
    bt->left_off->stack[0] = new_node(bt,NULL);
  }
  bt->left_off->top=1;
  node = bt->head;
  bt->left_off->stack[bt->left_off->top++] = node;
  while(node->left) {
    node=node->left;
    bt->left_off->stack[bt->left_off->top++] = node;
  }
}
node = ret = bt->left_off->stack[--bt->left_off->top];
if(bt->left_off->top == 0) {  /* done, nothing to return */
  free(bt->left_off->stack);
  free(bt->left_off);
  bt->left_off = NULL;
  ret->right = bt->unused;
  bt->unused = ret;
  *cont = 0;
  return NULL;
}
if(node->right) {
  bt->left_off->top++;
  bt->left_off->stack[bt->left_off->top++] = node->right;
  node = node->right;
  while(node->left) {
    node=node->left;
    bt->left_off->stack[bt->left_off->top++] = node;
  }
  return ret->data;
}
parent = bt->left_off->stack[bt->left_off->top-1];
while(node == parent->right) {
  node = parent;
  parent = bt->left_off->stack[--bt->left_off->top-1];
}
return ret->data;
}

void **BT_list(void *bti)
{
/*
  allocate a list of the tree contents, and return it
*/
  BTREE *bt = (BTREE *) bti;
  void **ret;
  NODE *stack[STACK_SIZE];
  NODE *node;
  int top=0,n=0;

  if(bt->n < 1) return NULL;
  ret = (void **) malloc(bt->n*sizeof(void *));
  node =  bt->head;

next_left:
  stack[top++] = node;
  if(node->left) {
    node = node->left;
    goto next_left;
  }
next:
  ret[n++] = node->data;
  if(node->right) {
    node = node->right;
    goto next_left;
  }
  top--;
  while(top>0 && stack[top-1]->right == stack[top]) {
     top--;
  }
  if(top > 0) {
    node = stack[top-1];
    goto next;
  }
  return ret;
}

STATIC void BT_free_node(BTREE *bt, NODE *node)
{
/*
  Free a node and all it's subnodes
*/
  if(node->left) BT_free_node(bt,node->left);
  if(node->right) BT_free_node(bt,node->right);
  node->right = bt->unused;
  bt->unused = node;
}

void BT_flush(void *bti)
{
/*
  Remove all nodes from the tree
*/
  BTREE *bt = (BTREE *) bti;
  NODE *del;

  if(bt->head) BT_free_node(bt,bt->head);
  if(bt->left_off) {
    del = bt->left_off->stack[0];
    del->right = bt->unused;
    bt->unused = del;
    free(bt->left_off->stack);
    free(bt->left_off);
    bt->left_off = NULL;
  }
  bt->head = NULL;
  bt->n = 0;
}

void BT_free(void *bti)
{
/*
  Free the tree and everything in it
*/
  BTREE *bt = (BTREE *)bti;
  int i;
  BT_flush(bt);
  for(i=0; i<bt->nmemory; i++)
    free(bt->memory[i]);
  free(bt->memory);
  free(bt);
}


#ifdef BTREE_DEBUG

BT_dump_tree2(NODE *node)
{
 if(node->left) {
   printf("LEFT:\n");
   BT_dump_tree2(node->left);
 }
 printf("%d   -- balance=%d\n",*(int *) node->data,node->balance);
 if(node->right) {
   printf("RIGHT:\n");
   BT_dump_tree2(node->right);
 }
 printf("UP\n");
}
BT_dump_tree(BTREE *btree)
{
  printf("Dump of tree:\n");
  BT_dump_tree2(btree->head);
}
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define POOLSIZE 1000000
#define MIX 1000000
#define POOL2SIZE 500000
#define LOOP1 100
#define LOOP2 50


int mycomp(int *i1, int *i2) { return *i1 - *i2; }

main()
{
int pool[POOLSIZE];
int pool2[POOLSIZE];
int i,j,k,n,loop,outer;
void *btree = BT_new(mycomp);
struct timeval start,stop;
double d;

for(outer=0; 1 ; outer++) {
  for(i=0; i<POOLSIZE; i++) {
    pool[i]=i;
  }

  for(i=0; i<MIX; i++) {
    j= rand()%POOLSIZE;
    k= rand()%POOLSIZE;
    n = pool[j];
    pool[j] = pool[k];
    pool[k] = n;
  }
  printf("TREE %d\n",outer);

  gettimeofday(&start,NULL);
  for(i=0; i<POOLSIZE; i++) {
    BT_store(btree,pool+i);
  }
  gettimeofday(&stop,NULL);
  d = (double)(stop.tv_sec-start.tv_sec)+(double)(stop.tv_usec-start.tv_usec)/1000000.;
  printf("%d elements added in %lf seconds\n",BT_numentries(btree),d);
  printf("  = %lf per second\n",(double)BT_numentries(btree)/d);
  BT_verify(btree);
  printf("deleting/adding %d elements\n",POOL2SIZE);
/*
  Put together a list of nodes to delete, then add back in
*/
  for(i=0; i<POOLSIZE; i++) {
    pool2[i]=i;
  }

  for(loop = 0; loop < LOOP2; loop++) {
    printf("*"); fflush(stdout);
    for(i=0; i<MIX; i++) {
      j= rand()%POOL2SIZE;
      k= rand()%POOLSIZE;
      n = pool2[j];
      pool2[j] = pool2[k];
      pool2[k] = n;
    }
    gettimeofday(&start,NULL);
    for(i=0; i<POOL2SIZE; i++) {
       BT_delete(btree,pool+pool2[i]);
    }
    gettimeofday(&stop,NULL);
  d = (double)(stop.tv_sec-start.tv_sec)+(double)(stop.tv_usec-start.tv_usec)/1000000.;
  printf("%d elements deleted in %lf seconds\n",POOL2SIZE,d);
  printf("  = %lf per second\n",(double)POOL2SIZE/d);
    if(BT_verify(btree) == -1) exit(0);
    gettimeofday(&start,NULL);
    for(i=0; i<POOL2SIZE; i++) {
       BT_store(btree,pool+pool2[i]);
    }
    gettimeofday(&stop,NULL);
  d = (double)(stop.tv_sec-start.tv_sec)+(double)(stop.tv_usec-start.tv_usec)/1000000.;
  printf("%d elements restored in %lf seconds\n",POOL2SIZE,d);
  printf("  = %lf per second\n",(double)POOL2SIZE/d);
    if(BT_verify(btree) == -1) exit(0);
  }
  printf(" %d\n",BT_numentries(btree));
  BT_flush(btree);
}
printf("The tree holds %d entries\n",BT_numentries(btree));
}
#endif
