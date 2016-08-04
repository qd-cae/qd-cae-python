
#include <iostream>
#include <iomanip>
#include "IOUtility.h"

void IOUtility::loadbar(unsigned int x, unsigned int n, unsigned int w = 40){

  //if ( (x != n) && (x % (n/100+1) != 0) )
  //  return;

  /*
  float ratio  =  x/(float)n;
  unsigned int   c      =  ratio * w;

  cout << setw(3) << (int)(ratio*100) << "% [";
  for (unsigned int x=0; x<c; x++)
    cout << "=";
  for (unsigned int x=c; x<w; x++)
    cout << " ";
  cout << "] (" << x << "/" << n << ")\r" << flush;
  */
}


void IOUtility::loadbar_end(){

  cout << endl;

}
