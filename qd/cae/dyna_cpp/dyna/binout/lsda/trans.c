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

static int little_i = 1;
#define little_endian (*(char *)(&little_i))

/*
   Translation functions
*/
static void _i2_i2_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i;

  count *= 2;
  for(i=0; i<count; i+= 2) {
    out[i+1] = in[i];
    out[i] = in[i+1];
  }
}
static void _i2_i4_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    if(in[i2+1] & 0x80)            /* sign extension */
      out[i4] = out[i4+1] = 0xff;
    else
      out[i4] = out[i4+1] = 0;
    out[i4+2] = in[i2+1];
    out[i4+3] = in[i2];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    if(in[i2] & 0x80)            /* sign extension */
      out[i4+3] = out[i4+2] = 0xff;
    else
      out[i4+3] = out[i4+2] = 0;
    out[i4+1] = in[i2];
    out[i4  ] = in[i2+1];
  }
}
}
static void _i2_i8_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    if(in[i2+1] & 0x80)            /* sign extension */
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = 0xff;
    else
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = 0;
    out[i8+6] = in[i2+1];
    out[i8+7] = in[i2];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    if(in[i2] & 0x80)            /* sign extension */
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = 0xff;
    else
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = 0;
    out[i8+1] = in[i2];
    out[i8  ] = in[i2+1];
  }
}
}
static void _i4_i4_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i;

  count *= 4;
  for(i=0; i<count; i+= 4) {
    out[i+3] = in[i  ];
    out[i+2] = in[i+1];
    out[i+1] = in[i+2];
    out[i  ] = in[i+3];
  }
}
static void _i4_i8_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    if(in[i4+3] & 0x80)            /* sign extension */
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = 0xff;
    else
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = 0;
    out[i8+4] = in[i4+3];
    out[i8+5] = in[i4+2];
    out[i8+6] = in[i4+1];
    out[i8+7] = in[i4];
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    if(in[i4] & 0x80)            /* sign extension */
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = 0xff;
    else
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = 0;
    out[i8+3] = in[i4  ];
    out[i8+2] = in[i4+1];
    out[i8+1] = in[i4+2];
    out[i8  ] = in[i4+3];
  }
}
}
static void _i8_i8_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i;

  count *= 8;
  for(i=0; i<count; i+= 8) {
    out[i+7] = in[i  ];
    out[i+6] = in[i+1];
    out[i+5] = in[i+2];
    out[i+4] = in[i+3];
    out[i+3] = in[i+4];
    out[i+2] = in[i+5];
    out[i+1] = in[i+6];
    out[i  ] = in[i+7];
  }
}
/*
  Shortening conversions.  Watch for overflow
*/
static void _i2_i1_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2+1] & 0x80) {  /* input is negative */
      if((in[i2] & 0x80) && in[i2+1] == 0xff) { /* but not overflow */
        out[i] = in[i2];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i2]&0x80) || in[i2+1]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i2];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2  ] & 0x80) {  /* input is negative */
      if((in[i2+1] & 0x80) && in[i2] == 0xff) { /* but not overflow */
        out[i] = in[i2+1];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i2+1]&0x80) || in[i2]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i2+1];
    }
  }
}
}
static void _i4_i1_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4+3] & 0x80) {  /* input is negative */
      if((in[i4] & 0x80) && in[i4+1] == 0xff && in[i4+2] ==0xff && in[i4+3] == 0xff) { /* but not overflow */
        out[i] = in[i4];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i4]&0x80) || in[i4+1] || in[i4+2] || in[i4+3]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i4];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4  ] & 0x80) {  /* input is negative */
      if((in[i4+3] & 0x80) && in[i4+2] == 0xff && in[i4+1] ==0xff && in[i4] == 0xff) { /* but not overflow */
        out[i] = in[i4+3];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i4+3]&0x80) || in[i4+2] || in[i4+1] || in[i4  ]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i4+3];
    }
  }
}
}
static void _i8_i1_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8+7] & 0x80) {  /* input is negative */
      if((in[i8] & 0x80) && in[i8+1] == 0xff && in[i8+2] ==0xff && in[i8+3] == 0xff &&
          in[i8+4]==0xff && in[i8+5] == 0xff && in[i8+6] ==0xff && in[i8+7] == 0xff) { /* but not overflow */
        out[i] = in[i8];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i8]&0x80) || in[i8+1] || in[i8+2] || in[i8+3] ||
                   in[i8+4] || in[i8+5] || in[i8+6] || in[i8+7]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i8];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8  ] & 0x80) {  /* input is negative */
      if((in[i8+7] & 0x80) && in[i8+6] == 0xff && in[i8+5] == 0xff && in[i8+4] == 0xff &&
          in[i8+3] == 0xff && in[i8+2] == 0xff && in[i8+1] == 0xff && in[i8  ] == 0xff) { /* but not overflow */
        out[i] = in[i8+7];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i8+7]&0x80) || in[i8+6] || in[i8+5] || in[i8+4] ||
                     in[i8+3] || in[i8+2] || in[i8+1] || in[i8  ]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i8+7];
    }
  }
}
}
static void _i4_i2_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4+3] & 0x80) {  /* input is negative */
      if((in[i4+1] & 0x80) && in[i4+2] == 0xff && in[i4+3] == 0xff) { /* but not overflow */
        out[i2  ] = in[i4+1];
        out[i2+1] = in[i4  ];
      } else {
        out[i2  ] = 0x80;
        out[i2+1] = 0;
      }
    } else if((in[i4+1]&0x80) || in[i4+2] || in[i4+3]) { /* Positive overflow */
      out[i2  ] = 0x7f;
      out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i4+1];
      out[i2+1] = in[i4];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4  ] & 0x80) {  /* input is negative */
      if((in[i4+2] & 0x80) && in[i4+1] == 0xff && in[i4  ] == 0xff) { /* but not overflow */
        out[i2+1] = in[i4+2];
        out[i2  ] = in[i4+3];
      } else {
        out[i2+1] = 0x80;
        out[i2  ] = 0;
      }
    } else if((in[i4+2]&0x80) || in[i4+1] || in[i4  ]) { /* Positive overflow */
      out[i2+1] = 0x7f;
      out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i4+2];
      out[i2  ] = in[i4+3];
    }
  }
}
}
static void _i8_i2_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8+7] & 0x80) {  /* input is negative */
      if((in[i8+1] & 0x80) && in[i8+2] == 0xff && in[i8+3] == 0xff && in[i8+4] == 0xff &&
                              in[i8+5] == 0xff && in[i8+6] == 0xff && in[i8+7] == 0xff) { /* but not overflow */
        out[i2  ] = in[i8+1];
        out[i2+1] = in[i8  ];
      } else {
        out[i2  ] = 0x80;
        out[i2+1] = 0;
      }
    } else if((in[i8+1]&0x80) || in[i8+2] || in[i8+3] || in[i8+4] ||
                                 in[i8+5] || in[i8+6] || in[i8+7]) { /* Positive overflow */
      out[i2  ] = 0x7f;
      out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i8+1];
      out[i2+1] = in[i8];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8  ] & 0x80) {  /* input is negative */
      if((in[i8+6] & 0x80) && in[i8+5] == 0xff && in[i8+4] == 0xff && in[i8+3] == 0xff &&
                              in[i8+2] == 0xff && in[i8+1] == 0xff && in[i8  ] == 0xff) { /* but not overflow */
        out[i2+1] = in[i8+6];
        out[i2  ] = in[i8+7];
      } else {
        out[i2+1] = 0x80;
        out[i2  ] = 0;
      }
    } else if((in[i8+6]&0x80) || in[i8  ] || in[i8+1] || in[i8+2] ||
                                 in[i8+3] || in[i8+4] || in[i8+5]) { /* Positive overflow */
      out[i2+1] = 0x7f;
      out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i8+6];
      out[i2  ] = in[i8+7];
    }
  }
}

}
static void _i8_i4_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8+7] & 0x80) {  /* input is negative */
      if((in[i8+3] & 0x80) && in[i8+4] == 0xff && in[i8+5] == 0xff &&
                              in[i8+6] == 0xff && in[i8+7] == 0xff) { /* but not overflow */
        out[i4  ] = in[i8+3];
        out[i4+1] = in[i8+2];
        out[i4+2] = in[i8+1];
        out[i4+3] = in[i8  ];
      } else {
        out[i4  ] = 0x80;
        out[i4+1] = out[i4+2] = out[i4+3] = 0;
      }
    } else if((in[i8+3] & 0x80) || in[i8+4] || in[i8+5] || in[i8+6] || in[i8+7]) { /* Positive overflow */
      out[i4  ] = 0x7f;
      out[i4+1] = out[i4+2] = out[i4+3] = 0xff;
    } else {
      out[i4  ] = in[i8+3];
      out[i4+1] = in[i8+2];
      out[i4+2] = in[i8+1];
      out[i4+3] = in[i8  ];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8  ] & 0x80) {  /* input is negative */
      if((in[i8+4] & 0x80) && in[i8+3] == 0xff && in[i8+2] == 0xff &&
                              in[i8+1] == 0xff && in[i8  ] == 0xff) { /* but not overflow */
        out[i4+3] = in[i8+4];
        out[i4+2] = in[i8+5];
        out[i4+1] = in[i8+6];
        out[i4  ] = in[i8+7];
      } else {
        out[i4+3] = 0x80;
        out[i4+2] = out[i4+1] = out[i4  ] = 0;
      }
    } else if((in[i8+4]&0x80) || in[i8  ] || in[i8+1] ||
                                 in[i8+2] || in[i8+3]) { /* Positive overflow */
      out[i4+3] = 0x7f;
      out[i4+2] = out[i4+1] = out[i4  ] = 0xff;
    } else {
      out[i4+3] = in[i8+4];
      out[i4+2] = in[i8+5];
      out[i4+1] = in[i8+6];
      out[i4  ] = in[i8+7];
    }
  }
}
}

static void _i1_i2(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=i*2;
    out[i2+1] = in[i];
    if(in[i] & 0x80)
      out[i2] = 0xff;
    else
      out[i2] = 0;
  }
} else {
  for(i=0; i<count; i++) {
    i2=i*2;
    out[i2] = in[i];
    if(in[i] & 0x80)
      out[i2+1] = 0xff;
    else
      out[i2+1] = 0;
  }
}
}
static void _i1_i4(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    if(in[i] & 0x80)            /* sign extension */
      out[i4] = out[i4+1] = out[i4+2] = 0xff;
    else
      out[i4] = out[i4+1] = out[i4+2] = 0;
    out[i4+3] = in[i];
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    if(in[i] & 0x80)            /* sign extension */
      out[i4+3] = out[i4+2] = out[i4+1] = 0xff;
    else
      out[i4+3] = out[i4+2] = out[i4+1] = 0;
    out[i4  ] = in[i];
  }
}
}
static void _i1_i8(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i8=8*i;
    if(in[i] & 0x80)            /* sign extension */
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = out[i8+6] = 0xff;
    else
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = out[i8+6] = 0;
    out[i8+7] = in[i];
  }
} else {
  for(i=0; i<count; i++) {
    i8=8*i;
    if(in[i] & 0x80)            /* sign extension */
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = out[i8+1] = 0xff;
    else
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = out[i8+1] = 0;
    out[i8  ] = in[i];
  }
}
}
static void _i2_i4(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    if(in[i2] & 0x80)            /* sign extension */
      out[i4] = out[i4+1] = 0xff;
    else
      out[i4] = out[i4+1] = 0;
    out[i4+2] = in[i2];
    out[i4+3] = in[i2+1];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    if(in[i2+1] & 0x80)            /* sign extension */
      out[i4+3] = out[i4+2] = 0xff;
    else
      out[i4+3] = out[i4+2] = 0;
    out[i4+1] = in[i2+1];
    out[i4  ] = in[i2];
  }
}
}
static void _i2_i8(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    if(in[i2] & 0x80)            /* sign extension */
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = 0xff;
    else
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = 0;
    out[i8+6] = in[i2];
    out[i8+7] = in[i2+1];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    if(in[i2+1] & 0x80)            /* sign extension */
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = 0xff;
    else
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = 0;
    out[i8+1] = in[i2+1];
    out[i8  ] = in[i2];
  }
}
}
static void _i4_i8(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    if(in[i4  ] & 0x80)            /* sign extension */
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = 0xff;
    else
      out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = 0;
    out[i8+4] = in[i4  ];
    out[i8+5] = in[i4+1];
    out[i8+6] = in[i4+2];
    out[i8+7] = in[i4+3];
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    if(in[i4+3] & 0x80)            /* sign extension */
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = 0xff;
    else
      out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = 0;
    out[i8+3] = in[i4+3];
    out[i8+2] = in[i4+2];
    out[i8+1] = in[i4+1];
    out[i8  ] = in[i4  ];
  }
}
}
/*
  Shortening conversions.  Watch for overflow
*/
static void _i2_i1(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2] & 0x80) {  /* input is negative */
      if((in[i2+1] & 0x80) && in[i2  ] == 0xff) { /* but not overflow */
        out[i] = in[i2+1];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i2+1]&0x80) || in[i2]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i2+1];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2+1] & 0x80) {  /* input is negative */
      if((in[i2  ] & 0x80) && in[i2+1] == 0xff) { /* but not overflow */
        out[i] = in[i2  ];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i2  ]&0x80) || in[i2+1]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i2  ];
    }
  }
}
}
static void _i4_i1(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4  ] & 0x80) {  /* input is negative */
      if((in[i4+3] & 0x80) && in[i4+2] == 0xff && in[i4+1] ==0xff && in[i4  ] == 0xff) { /* but not overflow */
        out[i] = in[i4+3];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i4+3]&0x80) || in[i4+2] || in[i4+1] || in[i4  ]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i4+3];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4+3] & 0x80) {  /* input is negative */
      if((in[i4  ] & 0x80) && in[i4+1] == 0xff && in[i4+2] ==0xff && in[i4+3] == 0xff) { /* but not overflow */
        out[i] = in[i4  ];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i4  ]&0x80) || in[i4+1] || in[i4+2] || in[i4+3]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i4  ];
    }
  }
}
}
static void _i8_i1(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8  ] & 0x80) {  /* input is negative */
      if((in[i8+7] & 0x80) && in[i8+6] == 0xff && in[i8+5] ==0xff && in[i8+4] == 0xff &&
          in[i8+3]==0xff && in[i8+2] == 0xff && in[i8+1] ==0xff && in[i8  ] == 0xff) { /* but not overflow */
        out[i] = in[i8+7];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i8+7]&0x80) || in[i8+6] || in[i8+5] || in[i8+4] ||
                   in[i8+3] || in[i8+2] || in[i8+1] || in[i8  ]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i8+7];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8+7] & 0x80) {  /* input is negative */
      if((in[i8  ] & 0x80) && in[i8+1] == 0xff && in[i8+2] == 0xff && in[i8+3] == 0xff &&
          in[i8+4] == 0xff && in[i8+5] == 0xff && in[i8+6] == 0xff && in[i8+7] == 0xff) { /* but not overflow */
        out[i] = in[i8  ];
      } else {
        out[i] = 0x80;
      }
    } else if((in[i8  ]&0x80) || in[i8+1] || in[i8+2] || in[i8+3] ||
                     in[i8+4] || in[i8+5] || in[i8+6] || in[i8+7]) { /* Positive overflow */
      out[i] = 0x7f;
    } else {
      out[i] = in[i8  ];
    }
  }
}
}
static void _i4_i2(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4  ] & 0x80) {  /* input is negative */
      if((in[i4+2] & 0x80) && in[i4+1] == 0xff && in[i4  ] == 0xff) { /* but not overflow */
        out[i2  ] = in[i4+2];
        out[i2+1] = in[i4+3];
      } else {
        out[i2  ] = 0x80;
        out[i2+1] = 0;
      }
    } else if((in[i4+2]&0x80) || in[i4+1] || in[i4+0]) { /* Positive overflow */
      out[i2  ] = 0x7f;
      out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i4+2];
      out[i2+1] = in[i4+3];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4+3] & 0x80) {  /* input is negative */
      if((in[i4+1] & 0x80) && in[i4+2] == 0xff && in[i4+3] == 0xff) { /* but not overflow */
        out[i2+1] = in[i4+1];
        out[i2  ] = in[i4  ];
      } else {
        out[i2+1] = 0x80;
        out[i2  ] = 0;
      }
    } else if((in[i4+1]&0x80) || in[i4+2] || in[i4+3]) { /* Positive overflow */
      out[i2+1] = 0x7f;
      out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i4+1];
      out[i2  ] = in[i4  ];
    }
  }
}
}
static void _i8_i2(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8  ] & 0x80) {  /* input is negative */
      if((in[i8+6]&0x80) && in[i8+5] == 0xff && in[i8+4] == 0xff && in[i8+3] == 0xff &&
                            in[i8+2] == 0xff && in[i8+1] == 0xff && in[i8  ] == 0xff) { /* but not overflow */
        out[i2  ] = in[i8+6];
        out[i2+1] = in[i8+7];
      } else {
        out[i2  ] = 0x80;
        out[i2+1] = 0;
      }
    } else if((in[i8+6]&0x80) || in[i8+5] || in[i8+4] || in[i8+3] ||
                                 in[i8+2] || in[i8+1] || in[i8  ]) { /* Positive overflow */
      out[i2  ] = 0x7f;
      out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i8+6];
      out[i2+1] = in[i8+7];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8+7] & 0x80) {  /* input is negative */
      if((in[i8+1] & 0x80) && in[i8+2] == 0xff && in[i8+3] == 0xff && in[i8+4] == 0xff &&
                              in[i8+5] == 0xff && in[i8+6] == 0xff && in[i8+7] == 0xff) { /* but not overflow */
        out[i2+1] = in[i8+1];
        out[i2  ] = in[i8  ];
      } else {
        out[i2+1] = 0x80;
        out[i2  ] = 0;
      }
    } else if((in[i8+1]&0x80) || in[i8+7] || in[i8+6] || in[i8+5] ||
                                 in[i8+4] || in[i8+3] || in[i8+2]) { /* Positive overflow */
      out[i2+1] = 0x7f;
      out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i8+1];
      out[i2  ] = in[i8  ];
    }
  }
}

}
static void _i8_i4(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8  ] & 0x80) {  /* input is negative */
      if((in[i8+4] & 0x80) && in[i8+3] == 0xff && in[i8+2] == 0xff && 
                              in[i8+1] == 0xff && in[i8  ] == 0xff) { /* but not overflow */
        out[i4  ] = in[i8+4];
        out[i4+1] = in[i8+5];
        out[i4+2] = in[i8+6];
        out[i4+3] = in[i8+7];
      } else {
        out[i4  ] = 0x80;
        out[i4+1] = out[i4+2] = out[i4+3] = 0;
      }
    } else if((in[i8+4] & 0x80) || in[i8+3] || in[i8+2] || in[i8+1] || in[i8  ]) { /* Positive overflow */
      out[i4  ] = 0x7f;
      out[i4+1] = out[i4+2] = out[i4+3] = 0xff;
    } else {
      out[i4  ] = in[i8+4];
      out[i4+1] = in[i8+5];
      out[i4+2] = in[i8+6];
      out[i4+3] = in[i8+7];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8+7] & 0x80) {  /* input is negative */
      if((in[i8+3] & 0x80) && in[i8+4] == 0xff && in[i8+5] == 0xff &&
                              in[i8+6] == 0xff && in[i8+7] == 0xff) { /* but not overflow */
        out[i4+3] = in[i8+3];
        out[i4+2] = in[i8+2];
        out[i4+1] = in[i8+1];
        out[i4  ] = in[i8  ];
      } else {
        out[i4+3] = 0x80;
        out[i4+2] = out[i4+1] = out[i4  ] = 0;
      }
    } else if((in[i8+3]&0x80) || in[i8+7] || in[i8+6] ||
                                 in[i8+5] || in[i8+4]) { /* Positive overflow */
      out[i4+3] = 0x7f;
      out[i4+2] = out[i4+1] = out[i4  ] = 0xff;
    } else {
      out[i4+3] = in[i8+3];
      out[i4+2] = in[i8+2];
      out[i4+1] = in[i8+1];
      out[i4  ] = in[i8  ];
    }
  }
}
}
/*
  Now the unsigned conversions....
*/
static void _u2_u4_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    out[i4] = out[i4+1] = 0;
    out[i4+2] = in[i2+1];
    out[i4+3] = in[i2];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    out[i4+3] = out[i4+2] = 0;
    out[i4+1] = in[i2];
    out[i4  ] = in[i2+1];
  }
}
}
static void _u2_u8_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = 0;
    out[i8+6] = in[i2+1];
    out[i8+7] = in[i2];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = 0;
    out[i8+1] = in[i2];
    out[i8  ] = in[i2+1];
  }
}
}
static void _u4_u8_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = 0;
    out[i8+4] = in[i4+3];
    out[i8+5] = in[i4+2];
    out[i8+6] = in[i4+1];
    out[i8+7] = in[i4];
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = 0;
    out[i8+3] = in[i4  ];
    out[i8+2] = in[i4+1];
    out[i8+1] = in[i4+2];
    out[i8  ] = in[i4+3];
  }
}
}
/*
  Shortening conversions.  Watch for overflow
*/
static void _u2_u1_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2+1]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i2];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i2+1];
    }
  }
}
}
static void _u4_u1_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4+1] || in[i4+2] || in[i4+3]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i4];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4+2] || in[i4+1] || in[i4  ]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i4+3];
    }
  }
}
}
static void _u8_u1_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8+1] || in[i8+2] || in[i8+3] ||
       in[i8+4] || in[i8+5] || in[i8+6] || in[i8+7]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i8];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8+6] || in[i8+5] || in[i8+4] ||
       in[i8+3] || in[i8+2] || in[i8+1] || in[i8  ]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i8+7];
    }
  }
}
}
static void _u4_u2_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4+2] || in[i4+3]) { /* Overflow */
      out[i2  ] = out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i4+1];
      out[i2+1] = in[i4];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4+1] || in[i4  ]) { /* Overflow */
      out[i2+1] = out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i4+2];
      out[i2  ] = in[i4+3];
    }
  }
}
}
static void _u8_u2_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8+2] || in[i8+3] || in[i8+4] ||
       in[i8+5] || in[i8+6] || in[i8+7]) { /* Overflow */
      out[i2  ] = out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i8+1];
      out[i2+1] = in[i8];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8  ] || in[i8+1] || in[i8+2] ||
       in[i8+3] || in[i8+4] || in[i8+5]) { /* Overflow */
      out[i2+1] = out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i8+6];
      out[i2  ] = in[i8+7];
    }
  }
}

}
static void _u8_u4_swap(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8+4] || in[i8+5] || in[i8+6] || in[i8+7]) { /* Overflow */
      out[i4  ] = out[i4+1] = out[i4+2] = out[i4+3] = 0xff;
    } else {
      out[i4  ] = in[i8+3];
      out[i4+1] = in[i8+2];
      out[i4+2] = in[i8+1];
      out[i4+3] = in[i8  ];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8  ] || in[i8+1] ||
       in[i8+2] || in[i8+3]) { /* Overflow */
      out[i4+3] = out[i4+2] = out[i4+1] = out[i4  ] = 0xff;
    } else {
      out[i4+3] = in[i8+4];
      out[i4+2] = in[i8+5];
      out[i4+1] = in[i8+6];
      out[i4  ] = in[i8+7];
    }
  }
}
}

static void _u1_u2(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=i*2;
    out[i2+1] = in[i];
    out[i2] = 0;
  }
} else {
  for(i=0; i<count; i++) {
    i2=i*2;
    out[i2] = in[i];
    out[i2+1] = 0;
  }
}
}
static void _u1_u4(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    out[i4] = out[i4+1] = out[i4+2] = 0;
    out[i4+3] = in[i];
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    out[i4+3] = out[i4+2] = out[i4+1] = 0;
    out[i4  ] = in[i];
  }
}
}
static void _u1_u8(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i8=8*i;
    out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = out[i8+6] = 0;
    out[i8+7] = in[i];
  }
} else {
  for(i=0; i<count; i++) {
    i8=8*i;
    out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = out[i8+1] = 0;
    out[i8  ] = in[i];
  }
}
}
static void _u2_u4(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    out[i4] = out[i4+1] = 0;
    out[i4+2] = in[i2];
    out[i4+3] = in[i2+1];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;
    out[i4+3] = out[i4+2] = 0;
    out[i4+1] = in[i2+1];
    out[i4  ] = in[i2];
  }
}
}
static void _u2_u8(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = out[i8+4] = out[i8+5] = 0;
    out[i8+6] = in[i2];
    out[i8+7] = in[i2+1];
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;
    out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = out[i8+3] = out[i8+2] = 0;
    out[i8+1] = in[i2+1];
    out[i8  ] = in[i2];
  }
}
}
static void _u4_u8(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    out[i8] = out[i8+1] = out[i8+2] = out[i8+3] = 0;
    out[i8+4] = in[i4  ];
    out[i8+5] = in[i4+1];
    out[i8+6] = in[i4+2];
    out[i8+7] = in[i4+3];
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;
    out[i8+7] = out[i8+6] = out[i8+5] = out[i8+4] = 0;
    out[i8+3] = in[i4+3];
    out[i8+2] = in[i4+2];
    out[i8+1] = in[i4+1];
    out[i8  ] = in[i4  ];
  }
}
}
/*
  Shortening conversions.  Watch for overflow
*/
static void _u2_u1(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i2+1];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;

    if(in[i2+1]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i2  ];
    }
  }
}
}
static void _u4_u1(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4+2] || in[i4+1] || in[i4  ]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i4+3];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;

    if(in[i4+1] || in[i4+2] || in[i4+3]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i4  ];
    }
  }
}
}
static void _u8_u1(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8+6] || in[i8+5] || in[i8+4] ||
       in[i8+3] || in[i8+2] || in[i8+1] || in[i8  ]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i8+7];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i8=8*i;

    if(in[i8+1] || in[i8+2] || in[i8+3] ||
       in[i8+4] || in[i8+5] || in[i8+6] || in[i8+7]) { /* Overflow */
      out[i] = 0xff;
    } else {
      out[i] = in[i8  ];
    }
  }
}
}
static void _u4_u2(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i4;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4+1] || in[i4+0]) { /* Overflow */
      out[i2  ] = out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i4+2];
      out[i2+1] = in[i4+3];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i4=4*i;

    if(in[i4+2] || in[i4+3]) { /* Overflow */
      out[i2+1] = out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i4+1];
      out[i2  ] = in[i4  ];
    }
  }
}
}
static void _u8_u2(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i2,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8+5] || in[i8+4] || in[i8+3] ||
       in[i8+2] || in[i8+1] || in[i8  ]) { /* Overflow */
      out[i2  ] = out[i2+1] = 0xff;
    } else {
      out[i2  ] = in[i8+6];
      out[i2+1] = in[i8+7];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i2=2*i;
    i8=8*i;

    if(in[i8+7] || in[i8+6] || in[i8+5] ||
       in[i8+4] || in[i8+3] || in[i8+2]) { /* Overflow */
      out[i2+1] = out[i2  ] = 0xff;
    } else {
      out[i2+1] = in[i8+1];
      out[i2  ] = in[i8  ];
    }
  }
}

}
static void _u8_u4(void *pin,void *pout,int count)
{
  unsigned char *in = (unsigned char *)pin;
  unsigned char *out = (unsigned char *)pout;
  int i,i4,i8;

if(!little_endian) {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8+3] || in[i8+2] || in[i8+1] || in[i8  ]) { /* Overflow */
      out[i4  ] = out[i4+1] = out[i4+2] = out[i4+3] = 0xff;
    } else {
      out[i4  ] = in[i8+4];
      out[i4+1] = in[i8+5];
      out[i4+2] = in[i8+6];
      out[i4+3] = in[i8+7];
    }
  }
} else {
  for(i=0; i<count; i++) {
    i4=4*i;
    i8=8*i;

    if(in[i8+7] || in[i8+6] || in[i8+5] || in[i8+4]) { /* Overflow */
      out[i4+3] = out[i4+2] = out[i4+1] = out[i4  ] = 0xff;
    } else {
      out[i4+3] = in[i8+3];
      out[i4+2] = in[i8+2];
      out[i4+1] = in[i8+1];
      out[i4  ] = in[i8  ];
    }
  }
}
}
/*
  Real conversions here
*/
#if defined(_HAVE_R4) && defined(_HAVE_R8)

static void _r4_r8(void *pin,void *pout,int count)
{
  TYPE_R4 *in  = (TYPE_R4 *) pin;
  TYPE_R8 *out = (TYPE_R8 *) pout;
  int i;
  for(i=0; i<count; i++) {
    out[i] = (TYPE_R8) in[i];
  }
}
static void _r8_r4(void *pin,void *pout,int count)
{
  TYPE_R8 *in  = (TYPE_R8 *) pin;
  TYPE_R4 *out = (TYPE_R4 *) pout;
  int i;
  for(i=0; i<count; i++) {
    out[i] = (TYPE_R4) in[i];
  }
}
static void _r4_r8_swap(void *pin,void *pout,int count)
{
  unsigned char *in  = (unsigned char *) pin;
  TYPE_R8 *out = (TYPE_R8 *) pout;
  TYPE_R4 swap;
  unsigned char *cp = (unsigned char *) &swap;
  int i,i4;

  for(i=0; i<count; i++) {
    i4=4*i;
    cp[0] = in[i4+3];
    cp[1] = in[i4+2];
    cp[2] = in[i4+1];
    cp[3] = in[i4  ];
    out[i] = (TYPE_R8) swap;
  }
}
static void _r8_r4_swap(void *pin,void *pout,int count)
{
  unsigned char *in  = (unsigned char *) pin;
  TYPE_R4 *out = (TYPE_R4 *) pout;
  TYPE_R8 swap;
  unsigned char *cp = (unsigned char *) &swap;
  int i,i8;

  for(i=0; i<count; i++) {
    i8=8*i;
    cp[0] = in[i8+7];
    cp[1] = in[i8+6];
    cp[2] = in[i8+5];
    cp[3] = in[i8+4];
    cp[4] = in[i8+3];
    cp[5] = in[i8+2];
    cp[6] = in[i8+1];
    cp[7] = in[i8  ];
    out[i] = (TYPE_R4) swap;
  }
}
#else
static void _r4_r8(void *pin,void *pout,int count)
{
  unsigned char *in  = (unsigned char *) pin;
  unsigned char *out = (unsigned char *) pout;
  int i,i4,i8;
  int s,e,f;

if(!little_endian) {
  for(i=0; i<count; i++) {
     i4=4*i;
     i8=8*i;
     s = in[i4] & 0x80;
     e = (in[i4] & 0x7f)<<1 & (in[i4+1]>>7);
     f = (in[i4+1] & 0x7f) | in[i4+2] | in[i4+3];
     if(e==255) {
       if(f) {        /* NaN */
         out[i8  ] = out[i8+1] = out[i8+2] = 0xff;
       } else if(s) {   /* - infinity */
         out[i8  ] = 0xff;
         out[i8+1] = 0xf0;
         out[i8+2] = out[i8+3] =
         out[i8+4] = out[i8+5] = out[i8+6] = out[i8+7] = 0;
       } else {   /* + infinity */
         out[i8  ] = 0x7f;
         out[i8+1] = 0xf0;
         out[i8+2] = out[i8+3] =
         out[i8+4] = out[i8+5] = out[i8+6] = out[i8+7] = 0;
       }
     } else if(e==0) {   /* 0  -> round subnormal numbers to 0 for simplicity */
       out[i8  ] = out[i8+1] = out[i8+2] = out[i8+3] =
       out[i8+4] = out[i8+5] = out[i8+6] = out[i8+7] = 0;
     } else {
       e += (1023-127);
       out[i8  ] = s | ((e&0x07ff)>>4);
       out[i8+1] = (e&0x0f)<<4);
       out[i8+2] = out[i8+3] = out[i8+4] = 0;
       out[i8+5] = in[i4+1] & 0x7f;
       out[i8+6] = in[i4+2];
       out[i8+7] = in[i4+3];
    }
  }
} else {
  for(i=0; i<count; i++) {
     i4=4*i;
     i8=8*i;
     s = in[i4+3] & 0x80;
     e = (in[i4+3] & 0x7f)<<1 & (in[i4+2]>>7);
     f = (in[i4+2] & 0x7f) | in[i4+1] | in[i4  ];
     if(e==255) {
       if(f) {        /* NaN */
         out[i8+7] = out[i8+6] = out[i8+5] = 0xff;
       } else if(s) {   /* - infinity */
         out[i8+7] = 0xff;
         out[i8+6] = 0xf0;
         out[i8+5] = out[i8+4] =
         out[i8+3] = out[i8+2] = out[i8+1] = out[i8  ] = 0;
       } else {   /* + infinity */
         out[i8+7] = 0x7f;
         out[i8+6] = 0xf0;
         out[i8+5] = out[i8+4] =
         out[i8+3] = out[i8+2] = out[i8+1] = out[i8  ] = 0;
       }
     } else if(e==0) {   /* 0  -> round subnormal numbers to 0 for simplicity */
       out[i8  ] = out[i8+1] = out[i8+2] = out[i8+3] =
       out[i8+4] = out[i8+5] = out[i8+6] = out[i8+7] = 0;
     } else {
       e += (1023-127);
       out[i8+7] = s | ((e&0x07ff)>>4);
       out[i8+6] = (e&0x0f)<<4);
       out[i8+5] = out[i8+4] = out[i8+3] = 0;
       out[i8+2] = in[i4+2] & 0x7f;
       out[i8+1] = in[i4+1];
       out[i8  ] = in[i4  ];
    }
  }
}
}
 have to complete these functions:
 static void _r8_r4(void *pin,void *pout,int count)
 static void _r4_r8_swap(void *pin,void *pout,int count)
 static void _r8_r4_swap(void *pin,void *pout,int count)
#endif

_CF GetConversionFunction(IFile *ifile,LSDAType *typein, LSDAType *typeout)
{
  int i;
  i = 256*LSDASizeOf(typein)+LSDASizeOf(typeout);

  if((typein->name[0]=='R') != (typeout->name[0]=='R')) {
    printf("IOLIB Error: cannot convert between real and integer types\n");
    return NULL;
  }
  if(typeout->name[0]=='U' || typein->name[0]=='U') {  /* use unsigned routines */
    if(ifile->bigendian == little_endian) {
      switch (i) {
        case 0x0101:   /* u1->u1, same endness */
        case 0x0202:   /* u2->u2, same endness */
        case 0x0404:   /* u4->u4, same endness */
        case 0x0808:   /* u8->u8, same endness */
          return NULL;
        case 0x0102:   /* u1->u2, same endness */
          return (_CF)_u1_u2;
        case 0x0104:   /* u1->u4, same endness */
          return (_CF)_u1_u4;
        case 0x0108:   /* u1->u8, same endness */
          return (_CF)_u1_u8;
        case 0x0201:   /* u2->u1, same endness */
          return (_CF)_u2_u1;
        case 0x0204:   /* u2->u4, same endness */
          return (_CF)_u2_u4;
        case 0x0208:   /* u2->u8, same endness */
          return (_CF)_u2_u8;
        case 0x0401:   /* u4->u1, same endness */
          return (_CF)_u4_u1;
        case 0x0402:   /* u4->u2, same endness */
          return (_CF)_u4_u2;
        case 0x0408:   /* u4->u8, same endness */
          return (_CF)_u4_u8;
        case 0x0801:   /* u8->u1, same endness */
          return (_CF)_u8_u1;
        case 0x0802:   /* u8->u2, same endness */
          return (_CF)_u8_u2;
        case 0x0804:   /* u8->u4, same endness */
          return (_CF)_u8_u4;
      }
    } else {
      switch (i) {
        case 0x0101:   /* u1->u1, different endness */
          return NULL;
        case 0x0202:   /* u2->u2, different endness */
          return (_CF)_i2_i2_swap;
        case 0x0404:   /* u4->u4, different endness */
          return (_CF)_i4_i4_swap;
        case 0x0808:   /* u8->u8, different endness */
          return (_CF)_i8_i8_swap;
        case 0x0102:   /* u1->u2, different endness, but no swapping for u1 */
          return (_CF)_u1_u2;
        case 0x0104:   /* u1->u4, different endness, but no swapping for u1 */
          return (_CF)_u1_u4;
        case 0x0108:   /* u1->u8, different endness, but no swapping for u1 */
          return (_CF)_u1_u8;
        case 0x0201:   /* u2->u1, different endness */
          return (_CF)_u2_u1_swap;
        case 0x0204:   /* u2->u4, different endness */
          return (_CF)_u2_u4_swap;
        case 0x0208:   /* u2->u8, different endness */
          return (_CF)_u2_u8_swap;
        case 0x0401:   /* u4->u1, different endness */
          return (_CF)_u4_u1_swap;
        case 0x0402:   /* u4->u2, different endness */
          return (_CF)_u4_u2_swap;
        case 0x0408:   /* u4->u8, different endness */
          return (_CF)_u4_u8_swap;
        case 0x0801:   /* u8->u1, different endness */
          return (_CF)_u8_u1_swap;
        case 0x0802:   /* u8->u2, different endness */
          return (_CF)_u8_u2_swap;
        case 0x0804:   /* u8->u4, different endness */
          return (_CF)_u8_u4_swap;
      }
    }
  } else if(typeout->name[0] == 'I') {     /* integer routines */
    if(ifile->bigendian == little_endian) {
      switch (i) {
        case 0x0101:   /* i1->i1, same endness */
        case 0x0202:   /* i2->i2, same endness */
        case 0x0404:   /* i4->i4, same endness */
        case 0x0808:   /* i8->i8, same endness */
          return NULL;
        case 0x0102:   /* i1->i2, same endness */
          return (_CF)_i1_i2;
        case 0x0104:   /* i1->i4, same endness */
          return (_CF)_i1_i4;
        case 0x0108:   /* i1->i8, same endness */
          return (_CF)_i1_i8;
        case 0x0201:   /* i2->i1, same endness */
          return (_CF)_i2_i1;
        case 0x0204:   /* i2->i4, same endness */
          return (_CF)_i2_i4;
        case 0x0208:   /* i2->i8, same endness */
          return (_CF)_i2_i8;
        case 0x0401:   /* i4->i1, same endness */
          return (_CF)_i4_i1;
        case 0x0402:   /* i4->i2, same endness */
          return (_CF)_i4_i2;
        case 0x0408:   /* i4->i8, same endness */
          return (_CF)_i4_i8;
        case 0x0801:   /* i8->i1, same endness */
          return (_CF)_i8_i1;
        case 0x0802:   /* i8->i2, same endness */
          return (_CF)_i8_i2;
        case 0x0804:   /* i8->i4, same endness */
          return (_CF)_i8_i4;
      }
    } else {
      switch (i) {
        case 0x0101:   /* i1->i1, different endness */
          return NULL;
        case 0x0202:   /* i2->i2, different endness */
          return (_CF)_i2_i2_swap;
        case 0x0404:   /* i4->i4, different endness */
          return (_CF)_i4_i4_swap;
        case 0x0808:   /* i8->i8, different endness */
          return (_CF)_i8_i8_swap;
        case 0x0102:   /* i1->i2, different endness, but no swapping for i1 */
          return (_CF)_i1_i2;
        case 0x0104:   /* i1->i4, different endness, but no swapping for i1 */
          return (_CF)_i1_i4;
        case 0x0108:   /* i1->i8, different endness, but no swapping for i1 */
          return (_CF)_i1_i8;
        case 0x0201:   /* i2->i1, different endness */
          return (_CF)_i2_i1_swap;
        case 0x0204:   /* i2->i4, different endness */
          return (_CF)_i2_i4_swap;
        case 0x0208:   /* i2->i8, different endness */
          return (_CF)_i2_i8_swap;
        case 0x0401:   /* i4->i1, different endness */
          return (_CF)_i4_i1_swap;
        case 0x0402:   /* i4->i2, different endness */
          return (_CF)_i4_i2_swap;
        case 0x0408:   /* i4->i8, different endness */
          return (_CF)_i4_i8_swap;
        case 0x0801:   /* i8->i1, different endness */
          return (_CF)_i8_i1_swap;
        case 0x0802:   /* i8->i2, different endness */
          return (_CF)_i8_i2_swap;
        case 0x0804:   /* i8->i4, different endness */
          return (_CF)_i8_i4_swap;
      }
    }
  } else {   /* use REAL routines */
    if(ifile->bigendian == little_endian) {
      switch (i) {
        case 0x0404:   /* r4->r4, same endness */
        case 0x0808:   /* r8->r8, same endness */
          return NULL;
        case 0x0408:   /* r4->r8, same endness */
          return (_CF)_r4_r8;
        case 0x0804:   /* r8->r4, same endness */
          return (_CF)_r8_r4;
      }
    } else {
      switch (i) {
        case 0x0404:   /* r4->r4, different endness */
          return (_CF)_i4_i4_swap;
        case 0x0808:   /* r8->r8, different endness */
          return (_CF)_i8_i8_swap;
        case 0x0408:   /* r4->r8, different endness */
          return (_CF)_r4_r8_swap;
        case 0x0804:   /* r8->r4, different endness */
          return (_CF)_r8_r4_swap;
      }
    }
  }
  return NULL;  /* can't get here, but SGI complains....*/
}
