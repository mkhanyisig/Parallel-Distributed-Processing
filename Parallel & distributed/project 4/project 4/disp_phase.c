/* disp_clocks.c
 * Main program that uses XWindows to visualize the output of
 * sim_slow. 
*/

#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include "my_timing.h"
#include "my_math.h"
#include "phase_io.h"

/*
  Global Variables determining window and display
*/
Display     *curDisplay;
Window     curWindow;
GC           curGC; /* graphics context */
XFontStruct *defaultFont;
int          defdepth;
int          bytesPerPixel;

#define INTERACTIVE_MODE 1
#define MOVIE_MODE       2

/*
  Global Variables for SCN-simulation data
*/
PhaseSimulationStruct *sim;
int curFrame = 0;
int gridX = 0;
float    *Px;
float    *Py;
float    *angles;

// The main drawing function is drawFigure.
void drawFigure();

// Set up a window.
int initWindow( long width, long height ) {
  int defScreen;
  XSetWindowAttributes wAttr;
  XGCValues gcValues;
  char buffer[64] = "Graphics";
  XTextProperty xtp = {(unsigned char *)buffer, 125, 8, strlen(buffer)};

  /*
   * connect to the X server.  uses the server specified in the
   * DISPLAY environment variable
   */
  curDisplay = XOpenDisplay((char *) NULL);
  if ((Display *) NULL == curDisplay) {
    fprintf(stderr, "Clock Display:  could not open display.\n");
      exit(-1);
  }
   
  /*
   * begin to create a window
   */
  defdepth = DefaultDepth(curDisplay,0);
  bytesPerPixel = defdepth/8;
  bytesPerPixel = bytesPerPixel == 3 ? 4 : bytesPerPixel;

  defScreen = DefaultScreen(curDisplay);

  curWindow = XCreateWindow(curDisplay, DefaultRootWindow(curDisplay),
                            10, 10, width, height, 0, 
                            defdepth, InputOutput, 
                            DefaultVisual(curDisplay, defScreen),
                            0, &wAttr);


  /*
   * request mouse button and keypress events
   */
  wAttr.event_mask = ButtonPressMask | KeyPressMask | ExposureMask;
  XChangeWindowAttributes(curDisplay, curWindow, CWEventMask, &wAttr);

  /*
   * force it to appear on the screen
   */
  XSetWMName(curDisplay, curWindow, &xtp);
  XMapWindow(curDisplay, curWindow);

  /*
   * create a graphics context.  this stores a drawing state; stuff like
   * current color and line width.  this gc is for drawing into our
   * window. 
   */
  curGC = XCreateGC(curDisplay, curWindow, 0, &gcValues);
  defaultFont = XQueryFont(curDisplay, XGContextFromGC(curGC));

  XSetWindowColormap( curDisplay,
                      curWindow,
                      DefaultColormapOfScreen(DefaultScreenOfDisplay(curDisplay)));

  return(bytesPerPixel);
}

void refreshWindow()
{
  // refresh the window here
  drawFigure();
  return;
}

void closeWindow(long nFrames)
{
  XCloseDisplay(curDisplay);
  return;
}

// Computes the width and height of the string's bounding box.
// Returns the width, font ascent, and font descent in the
// appropriate variables.
void getStringBoundingBox(char *string, int len, int *width, int *ascent, int *descent) {
  int dir;
  XCharStruct char_info;
  XTextExtents(defaultFont,string,len,&dir,ascent,descent,&char_info); 
  *width = (int)char_info.width;
} // end getStringBoundingBox

// draw box with lower left-hand corner at (x0, y0)
void box( int x0, int y0, int w, int h ) {
	
  XDrawLine( curDisplay, curWindow, curGC, x0, y0, x0 + w, y0 );
  XDrawLine( curDisplay, curWindow, curGC, x0 + w, y0, x0 + w, y0 - h );
  XDrawLine( curDisplay, curWindow, curGC, x0 + w, y0 - h, x0, y0 - h);
  XDrawLine( curDisplay, curWindow, curGC, x0, y0 - h, x0, y0 );

  return;
} // end box

// Draw a clock (well, really its an arrow)
void clock(int x0, int y0, int d, float angle ) {
	
  int xf = x0 + d * cos( angle );
  int yf = y0 - d * sin( angle );

  XDrawLine( curDisplay, curWindow, curGC, x0, y0, xf, yf );

  if( d > 5 ) {
    int xa = x0 + (d-3) * cos( angle - 0.2 );
    int ya = y0 - (d-3) * sin( angle - 0.2 );

    XDrawLine( curDisplay, curWindow, curGC, xf, yf, xa, ya );

    xa =  x0 + (d-3) * cos( angle + 0.2 );
    ya =  y0 - (d-3) * sin( angle + 0.2 );

    XDrawLine( curDisplay, curWindow, curGC, xf, yf, xa, ya );
  }

  return;
} // end clock

void drawAllClocks(int width, int height, int xoff, int yoff) {
  if (sim == NULL)
    return;

  float minx = -0.5;
  float maxx = gridX - 0.5;
  float miny = -0.5;
  float maxy = gridX - 0.5;
  int   pad      = 20;
  int   clockSize = 15;
  int   strpad   = 5;
  float xscale   = (float)(width-2*pad)/(maxx-minx);
  float yscale   = (float)(height-2*pad)/(maxy-miny);
  int i, x_p, y_p;

  XSetWindowBackground( curDisplay, curWindow, 0x00FFFFFF );
  XClearWindow( curDisplay, curWindow );

  int xzero_p = xoff+pad+(int)((0.0-minx)*xscale);
  int yzero_p = yoff+pad+(int)((maxy-0.0)*yscale);
  int xstart_p = xoff;
  int xstop_p  = xoff+width;
  int ystart_p = yoff;
  int ystop_p  = yoff+height;
  int xmin_p  = xoff+pad+(int)((minx-minx)*xscale);
  int xmax_p  = xoff+pad+(int)((maxx-minx)*xscale);
  int ymin_p  = yoff+pad+(int)((maxy-miny)*yscale);
  int ymax_p  = yoff+pad+(int)((maxy-maxy)*yscale);

  // Draw clock in a medium green color
  XSetForeground( curDisplay, curGC, 0x0011AA33 );
  for (i=0; i<sim->Nx; i++) {
    clock(xoff+pad+(int)((Px[i]-minx)*xscale),
         yoff+pad+(int)((maxy-Py[i])*yscale),
         clockSize, angles[i]);
  }

} // end drawAllClocks

void drawFigure() {
  // Determine how large the window is
  Window root;
  int x_ret, y_ret;
  unsigned int width, height, bw, dr;
  XGetGeometry(curDisplay, curWindow, &root, &x_ret, &y_ret,
               &width, &height, &bw, &dr );

  const int title_height = 40;

  int xoff = 0;
  int yoff = title_height;
  drawAllClocks(width-xoff, height-yoff, xoff, yoff);
  
} // end drawFigure

// free space allocated for global variables
void cleanup() {
  if (sim) {
    free( sim->periods );
    free( sim->phases_RT );
  }
  if (Px)
    free(Px);
  if (Py)
    free(Py);
  if (angles)
    free(angles);
} // end cleanup

float getAngle( float phase_RT, float period ) {
    return M_PI/2.0 - phase_RT / period * 2 * M_PI;
}

int displaySimulation(char *filename, int stride, double secondsPerFrame) {
  Bool   done = False;
  XEvent curEvent;
  char   c;
  long   event_mask = 0xFFFFFFFF;
  long   nBuf = 2;
  int    mode = MOVIE_MODE; 
  double t1, t2;

  /* initialize the window  */
  initWindow( 600, 400 );

  /* initialize your models (read them from files) here */
  sim = readPhaseSimulationFile( filename );
  Px = malloc(sizeof(float)*sim->Nx);
  Py = malloc(sizeof(float)*sim->Nx);
  angles = malloc(sizeof(float)*sim->Nx);
  gridX = (int)ceil( sqrt( (double)sim->Nx ) );
  int i;
  for (i = 0; i < sim->Nx; i++) {
    Px[i] = (float) (i % gridX);
    Py[i] = (float) (gridX - 1 -  i / gridX);
    angles[i] = getAngle( sim->phases_RT[i], sim->periods[i] );
  }
//   dumpPhaseSimulation( sim );
  t1 = get_time_sec();
  
  while(!done) {

    /** Draw stuff here **/
    if (mode == MOVIE_MODE) {
      t2 = get_time_sec();
      if (t2-t1 > secondsPerFrame && curFrame+stride < sim->Nt) {
        curFrame+= stride;
        for (i = 0; i < sim->Nx; i++) {
          angles[i] = getAngle( sim->phases_RT[sim->Nx*curFrame+i], sim->periods[i] );
        }
        drawFigure();
        t1 = t2;
      }
    }

    usleep( 33 ); // sleep a little

    // check for events
    if(XCheckWindowEvent(curDisplay, curWindow, event_mask, &curEvent)) {

      switch (curEvent.type) {

      case KeyPress:
        /*
           access string using XLookupString(*event,*char,numChars,NULL,NULL)
        */
        if(XLookupString((XKeyEvent *)&curEvent, &c, 1, NULL, NULL) == 1) {
          switch(c) {
          case 'n':
            if (mode == INTERACTIVE_MODE) {
                curFrame+=stride;
                for (i = 0; i < sim->Nx; i++) {
                  angles[i] = getAngle( sim->phases_RT[sim->Nx*curFrame+i], sim->periods[i] );
                }
               drawFigure();
            }
            break;
          case 'i':
             mode = INTERACTIVE_MODE;
             break;
          case 'm':
             mode = MOVIE_MODE;
             break;
          case 'q':
	    // quit
	    done = True;
	    break;
          default:
            break;
          }
        }
        break;

      case Expose:
        refreshWindow();
        break;

      default:
       break;
      } // end switch over event type
    } // end if event
  } // end while

  closeWindow(nBuf);
  cleanup();
  return(0);
} // end displaySimulation

/* Main */
int main(int argc, char *argv[]) {
  char *filename;
  double secondsPerFrame = 0.01;
  int stride = 10;

  if (argc < 2) {
    printf("Usage: ./disp_phase <filename> [<secondsPerFrame>]\n");
    printf("       <filename> should be a .phs file\n");
    printf("       <skip> number of time steps to skip between frames (defaults to 10)\n");
    return;
  }

  filename = argv[1];

  if (argc > 2)
    stride = atoi(argv[2]);

  printf("Press 'q' to close window\n");
  displaySimulation(filename, stride, secondsPerFrame);
} // end main


