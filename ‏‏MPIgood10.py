import numpy

from matplotlib import pyplot, cm
import matplotlib.pyplot as plt

import    matplotlib
import matplotlib.pyplot as plt
import math
from mpi4py import MPI

import pandas as pd
import sys
import seaborn as sns 
import time
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
import scipy.interpolate as il

from mpl_toolkits.mplot3d import Axes3D

import random

from mpl_toolkits.mplot3d import Axes3D ##library for 3d projection plots
 
def firstdomain():
  nx = 31
  ny = 31
  nz = 31
  nt = 17
  nu = .05
  dx = 2 / (nx - 1)
  dy = 2 / (ny - 1)
  dz = 2 / (nz - 1)
  sigma = .25
  dt = sigma * dx * dy / nu

  x = numpy.linspace(0, 2, nx)
  y = numpy.linspace(0, 2, ny)

 

  u = numpy.ones((ny, nx))  # create a 1xn vector of 1's
  un = numpy.ones((ny, nx))

  ###Assign initial conditions
  # set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
  u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2  

  fig = pyplot.figure()
  ax = fig.gca(projection='3d')
  X, Y = numpy.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.viridis,
          linewidth=0, antialiased=False)

  ax.set_xlim(0, 2)
  ax.set_ylim(0, 2)
  ax.set_zlim(1, 2.5)

  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$');

def diffuse(nt):
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2  
    
    for n in range(nt + 1): 
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)
  
 
    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$xx$')
    ax.set_ylabel('$yy$');
 

def diffuse2(nt):
 
    nx = 31
    ny = 31
    nz = 31
    nt = 17
    nu = .05
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dz = 2 / (nz - 1)
    sigma = .25
    dt = sigma * dx * dy / nu

    x = numpy.linspace(0, 2, nx)
    y = numpy.linspace(0, 2, ny)

 

    u = numpy.ones((ny, nx))  # create a 1xn vector of 1's
    un = numpy.ones((ny, nx))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
 
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2  
    
    for n in range(nt + 1): 
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)
  
 
    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$xx$')
    ax.set_ylabel('$yy$');

def diffuse3(nt):
 
    nx = 31
    ny = 31
    nz = 31
    nt = 17
    nu = .05
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dz = 2 / (nz - 1)
    sigma = .25
    dt = sigma * dx * dy / nu

    x = numpy.linspace(0, 2, nx)
    y = numpy.linspace(0, 2, ny)

 

    u = numpy.ones((ny, nx))  # create a 1xn vector of 1's
    un = numpy.ones((ny, nx))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
 
    u[int(0 / dy):int(1 / dy + 1),int(0 / dx):int(1 / dx + 1)] = 2  
    
    for n in range(nt + 1): 
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    
    X, Y = numpy.meshgrid(x, y)
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)
  
 
    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$xx$')
    ax.set_ylabel('$yy$');
 

def diffuse4(nt):
 
    nx = 31
    ny = 31
    nz = 31
    nt = 17
    nu = .05
    largx = 3  #Domain length. 4 or 11  or 2
    largy = 3
 

    dx = largx / (nx - 1) #delta x ist der Abstand zwischen einem beliebigen Paar benachbarter Gitterpunkte
    dy = largy / (ny - 1) #delta y
 
    sigma = .25
    dt = sigma * dx * dy / nu

    x = numpy.linspace(0, 2, nx)
    y = numpy.linspace(0, 2, ny)

 

    u = numpy.ones((nx, ny))  # create a 1xn vector of 1's
    un = numpy.ones((nx, ny))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
 
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = largx  
    
    for n in range(nt + 1): 
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:],  cmap=pyplot.jet()) 
  
 
    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$xx$')
    ax.set_ylabel('$yy$'); 

def diffuse43d(nt):
 
    nx = 31
    ny = 31
    nz = 31
    nt = 17
    nu = .05
    largx = 4  #Domain length. 4 or 11  or 2
    largy = 4
    largz = 4
 

    dx = largx / (nx - 1) #delta x ist der Abstand zwischen einem beliebigen Paar benachbarter Gitterpunkte
    dy = largy / (ny - 1) #delta y
    dz = largy / (nz - 1) #delta y
 
    sigma = .25
    dt = sigma * dx * dy * dz / nu

   
 

    uu = numpy.ones((nx, ny,nz))  # create a 1xn vector of 1's
    uunn = numpy.ones((nx, ny,nz))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
 
    uu[int(0 / dy):int(1 / dy + 1),int(0 / dx):int(1 / dx + 1) ,int(0 / dz):int(1 / dz + 1)] =1000  
    
    for n in range(nt + 1): 
        un = uu.copy()
        uunn = uu.copy()                                    
        uu[1:-1,1:-1,1:-1] = (uunn[1:-1,1:-1,1:-1] + 
                        nu*dt / dy**2 *                                       
                         uunn[1:-1, 2:,1:-1] - 2 * uunn[1:-1, 1:-1,1:-1] + uunn[1:-1, 0:-2,1:-1]) 
        uu[0,:,:] =  1
        uu[-1,:,:] = 1
        uu[:,0,:] =  1
        uu[:,-1,:] = 1
        uu[:,:,0] =  1
        uu[:,:,-1] = 1

    
  
    
 
    x4 = numpy.linspace(0,1,nx)   
    y4 = numpy.linspace(0,1,nx) 
    z4 = numpy.linspace(0,1,nx) 
 

    X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(X4, Y4, Z4, c=uu, cmap=cm.viridis,
        linewidth=0, antialiased=True)
    
    ax.set_xlabel('$xx$')
    ax.set_ylabel('$yy$'); 
    fig.colorbar(img)

 

    fig1 = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    img1 = plt.scatter(X4, Y4, nt , c=uu, cmap="plasma"  )
    ax.set_xlabel('$xx$')
    ax.set_ylabel('$yy$'); 
    fig1.colorbar(img1)

def diffuse53d(nt):
 
    nx = 31
    ny = 31
    nz = 31
    nt = 31
    nu = .05
    largx =100  #Domain length. 4 or 11  or 2
    largy = 100
    largz = 100
 

    dx = largx / (nx - 1) #delta x ist der Abstand zwischen einem beliebigen Paar benachbarter Gitterpunkte
    dy = largy / (ny - 1) #delta y
    dz = largy / (nz - 1) #delta y
 
    sigma = .25
    dt = sigma * dx * dy * dz / nu

   
 

    uu = numpy.ones((nx, ny,nz))  # create a 1xn vector of 1's
    uunn = numpy.ones((nx, ny,nz))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
 
    uu[int(0 / dy):int(1 / dy + 1),int(0 / dx):int(1 / dx + 1) ,int(0 / dz):int(1 / dz + 1)] = 200
    
    for n in range(nt + 1): 
        un = uu.copy()
        uunn = uu.copy()                                    
        uu[1:-1,1:-1,1:-1] =170
 
 
        uu[0,:,:] =  255
        uu[-1,:,:] = 155
        uu[:,0,:] =  100
        uu[:,-1,:] = 22
        uu[:,:,0] =  0
        uu[:,:,-1] = 50  
 
 
    w2_uno = numpy.reshape(uu,-1)   
    x4 = numpy.linspace(0,nx,nx)   
    y4 = numpy.linspace(0,0.1,nx) 
    z4 = numpy.linspace(0,0.1,nx) 

    X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
 
    img = ax.scatter(X4, Y4, Z4, c=w2_uno, cmap=cm.viridis,
        linewidth=0, antialiased=True)
    fig.colorbar(img)
 
  

def diff3d(tt):
    
    w2 = numpy.ones((kx,ky,kz))*Tin         
    wn2 = numpy.ones((kx,ky,kz))*Tin
   
    for k in range(tt+2):
        wn2 = w2.copy()
        w2[1:-1,1:-1,1:-1] = (wn2[1:-1,1:-1,1:-1] + 
                        kapp*dt4 / dy4**2 *                                       
                        (wn2[1:-1, 2:,1:-1] - 2 * wn2[1:-1, 1:-1,1:-1] + wn2[1:-1, 0:-2,1:-1]) +  
                        kapp*dt4 / dz4**2 *                                       
                        (wn2[1:-1,1:-1,2:] - 2 * wn2[1:-1, 1:-1,1:-1] + wn2[1:-1, 1:-1,0:-2]) +
                        kapp*dt4 / dx4**2 *
                        (wn2[2:,1:-1,1:-1] - 2 * wn2[1:-1, 1:-1,1:-1] + wn2[0:-2, 1:-1,1:-1]))

        #Neumann boundary (dx=dy=dz for the time)
        w2[0,:,:] =   w2[0,:,:] + 2*kapp* (dt4/(dx4**2)) * (w2[1,:,:] - w2[0,:,:] - qq1 * dx4/kapp)
        w2[-1,:,:] =  w2[-1,:,:] + 2* kapp*(dt4/(dx4**2)) * (w2[-2,:,:] - w2[-1,:,:] + qq2 * dx4/kapp)
        w2[:,0,:] =   w2[:,0,:] + 2*kapp* (dt4/(dx4**2)) * (w2[:,1,:] - w2[:,0,:] - qq3 * dx4/kapp)
        w2[:,-1,:] =  w2[:,-1,:] + 2*kapp* (dt4/(dx4**2)) * (w2[:,-2,:] - w2[:,-1,:] + qq4 * dx4/kapp)
        w2[:,:,0] =   w2[:,:,0] + 2 *kapp* (dt4/(dx4**2)) * (w2[:,:,-1] - w2[:,:,0] - qq5 * dx4/kapp)
        w2[:,:,-1] =   w2[:,:,-1] + 2 *kapp* (dt4/(dx4**2)) * (w2[:,:,-2] - w2[:,:,-1] + qq6 * dx4/kapp)
       
    w2[1:,:-1,:-1] = numpy.nan    #We'll only plot the "outside" points.
    w2_uno = numpy.reshape(w2,-1)    

    #Plotting
    fig = pyplot.figure()
    X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(X4, Y4, Z4, c=w2_uno, cmap=pyplot.jet())
    fig.colorbar(img)
 

#diffuse(10)
#diff3d(30)

 
#this is  true
def computeNexttemp(itra):
 
      # Index variables 
      # x0,  x,  size_x,  size_y,  size_z,  dt,  hx,  hy,     hz,  r, me,  xs,ys, zs,  xe, ye,  ze,  k0

      #Factors for the stencil  
      size_x = 4; #2 or 1 or 4 
      size_y = 4; #2 or 1 or 
      size_z = 4; #2 or 1 4 
      x_domains = itra;#4; 
      y_domains =itra; #4;
      z_domains =itra; #4;
      size_x_glo = size_x + 2;
      size_y_glo = size_y + 2;
      size_z_glo = size_z + 2;
      hx = 1.0 /  (size_x_glo);
      hy = 1.0 /  (size_y_glo);
      hz = 1.0 / (size_z_glo);
      dt = 48.0e-3;
      k0 = 260;
      #  Local    Variable for calculating the error
      # The template of the explicit operator for the pressing equation on a regular rectangular grid using a seven-point finite
      #  difference   Scheme in the room is:
 

      diagx = -2.0 + hx * hx / (3 * k0 * dt);
      diagy = -2.0 + hy * hy / (3 * k0 * dt); 
      diagz = -2.0 + hz * hz / (3 * k0 * dt);
      weightx = k0 * dt / (hx * hx);
      weighty = k0 * dt / (hy * hy);
      weightz = k0 * dt / (hz * hz);

      # Do an explicit update at the points within the domain 
      zse =itra 
      yse =itra 
      ise =itra 
 

      x0 = numpy.ones((zse,yse,ise)) 
      x = numpy.ones((zse,yse,ise))
    
      for k in range(1,zse -1): 
        for j in range(1,yse - 1):  
            for i in range(1,ise - 1):  
                x[i][j][k] =1

      w2_uno = numpy.reshape(x,-1)    
      x4 = numpy.linspace(0,1,x_domains)   
      y4 = numpy.linspace(0,1,y_domains)
      z4 = numpy.linspace(0,1,z_domains)



      fig = pyplot.figure()
      X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)
      ax = fig.add_subplot(111, projection='3d')
      img = ax.scatter(X4, Y4, Z4, c=w2_uno, cmap=cm.viridis,
        linewidth=0, antialiased=True)
                      #cmap=pyplot.jet())
      #rstride=1, cstride=1, cmap=cm.viridis,         linewidth=0, antialiased=True)
      fig.colorbar(img)
    

      # Copy the calculated value back: x0 (n) <- x (n) and calculate the 2_Norm of the '' residual '  
 
#Defining the function  .
def diffusetemp2(nt):
 

    uu = numpy.ones((nx, ny,nz))  # create a 1xn vector of 1's
    uunn = numpy.ones((nx, ny,nz))
 
    x4 = numpy.linspace(0,1,nx)   
    y4 = numpy.linspace(0,1,ny)
    z4 = numpy.linspace(0,1,nz)

    uu[int(.5 / dy):int(1 / dy + 1) , int(.5 / dx):int(1 / dx + 1) , int(.5 / dz):int(1 / dz + 1) ] = 3  
    
    for n in range(nt + 2): 
        uunn = uu.copy()                                    
        uu[1:-1,1:-1,1:-1] = (uunn[1:-1,1:-1,1:-1] + 
                        nu*dt / dy**2 *                                       
                        (uunn[1:-1, 2:,1:-1] - 2 * uunn[1:-1, 1:-1,1:-1] + uunn[1:-1, 0:-2,1:-1]) +  
                        nu*dt / dz**2 *                                       
                        (uunn[1:-1,1:-1,2:] - 2 * uunn[1:-1, 1:-1,1:-1] + uunn[1:-1, 1:-1,0:-2]) +
                        nu*dt / dx**2 *
                        (uunn[2:,1:-1,1:-1] - 2 * uunn[1:-1, 1:-1,1:-1] + uunn[0:-2, 1:-1,1:-1]))
        uu[0,:,:] =  1
        uu[-1,:,:] = 1
        uu[:,0,:] =  1
        uu[:,-1,:] = 1
        uu[:,:,0] =  1
        uu[:,:,-1] =  1

 
    uu[1:,:-1,:-1] = numpy.nan    #We'll only plot the "outside" points.
    w2_uno = numpy.reshape(uu,-1)    

 

    #Plotting
    fig = pyplot.figure()
    X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(X4, Y4, Z4, c=w2_uno, cmap=pyplot.jet())
    fig.colorbar(img)
 
 
#Defining the function  .
def diffusetemp1(nt):
 

    nx = 35   # ist die Anzahl der gewünschten Gitterpunkte 
    ny = 35   # 
    nz = 35   # 

    nt =100      #ist die Anzahl der Zeitschritte, die wir berechnen möchten 
    nu = 260      # Diffusionkoeffizient
    cin = 25     # anfang Konzentrationen

    ak= 13.3     # anfang druck
    ek= 5.3      # end druck

    largx = 4  #Domain length.
    largy = 4
    largz = 4

    dx = largx / (nx - 1) #delta x ist der Abstand zwischen einem beliebigen Paar benachbarter Gitterpunkte
    dy = largy / (ny - 1) #delta y
    dz = largz / (nz - 1) #delta z

    #delta dt=dx*dx/8D (dx=dy=dz)Stabilitätsbedingung 
    # D= 260 Diffusionkoeffizient
    dt =0.008  # in seunde 


    x = numpy.linspace(0, largx, nx)
    y = numpy.linspace(0, largy, ny)
    z = numpy.linspace(0, largz, nz)
    # Konzentrationen array
    uu = numpy.ones((ny, nx , nz))*cin  
    uunn = numpy.ones((ny, nx, nz))*cin


    ###Assign initial conditions
    # set hat function I.C. : u(0<=x<=35 && 0<=y<=35 && 0<=z<=35) is 70(largx,y,z)
    uu[int(0 / dy):int(35 / dy + 1),int(0 / dx):int(35 / dx + 1),int(0 / dz):int(35 / dz + 1)] = 70  
 
    
    for n in range(nt + 2): 
        uunn = uu.copy()                                    
        uu[1:-1,1:-1,1:-1] = (uunn[1:-1,1:-1,1:-1] + 
                        nu*dt / dy**2 *                                       
                        (uunn[1:-1, 2:,1:-1] - 2 * uunn[1:-1, 1:-1,1:-1] + uunn[1:-1, 0:-2,1:-1]) +  
                        nu*dt / dz**2 *                                       
                        (uunn[1:-1,1:-1,2:] - 2 * uunn[1:-1, 1:-1,1:-1] + uunn[1:-1, 1:-1,0:-2]) +
                        nu*dt / dx**2 *
                        (uunn[2:,1:-1,1:-1] - 2 * uunn[1:-1, 1:-1,1:-1] + uunn[0:-2, 1:-1,1:-1]))
        uu[0,:,:] =  35
        uu[-1,:,:] = 35
        uu[:,0,:] =  35
        uu[:,-1,:] = 35
        uu[:,:,0] =  35
        uu[:,:,-1] =  35

 
    uu[1:,:-1,:-1] = numpy.nan    #We'll only plot the "outside" points.
    w2_uno = numpy.reshape(uu,-1)    

  

 
    x4 = numpy.linspace(0,1,nx)   
    y4 = numpy.linspace(0,1,ny)
    z4 = numpy.linspace(0,1,nz)
 
    #Plotting
    fig = pyplot.figure()
    X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)
    ax = fig.add_subplot(222, projection='3d')
    img = ax.scatter(X4, Y4, Z4, c=w2_uno, cmap=pyplot.jet())
    fig.colorbar(img)

    
def diff4d(): 

    pyplot.rcParams["figure.figsize"] = [7.00, 3.50]
    pyplot.rcParams["figure.autolayout"] = True
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = numpy.random.standard_normal(100)
    y = numpy.random.standard_normal(100)
    z = numpy.random.standard_normal(100)
    c = numpy.random.standard_normal(100)
    img = ax.scatter(x, y, z, c=c, cmap='YlOrRd', alpha=1)

def tempfake():

  num_pts = 10000
  noise_frac = 0.07

  X1 = numpy.random.uniform(1, 51, num_pts).reshape(-1, 1)
  X2 = numpy.random.uniform(1, 51, num_pts).reshape(-1, 1)
  X3 = numpy.random.uniform(2, 51, num_pts).reshape(-1, 1)
  X4 = numpy.random.uniform(1, 51, num_pts).reshape(-1, 1)
  X5 = numpy.random.uniform(1, 51, num_pts).reshape(-1, 1)

  X = numpy.hstack((X1, X2, X3, X4, X5))

  Y = 1 * X1 + 2 * X2 + 3 * X3 + 4 * X4 + 5 * X5
  Y = Y.reshape(-1, 1)

  """ We add fake noise to our label """
  Ynoise = numpy.amax(Y) * noise_frac
  Y += numpy.random.uniform(-Ynoise, Ynoise, num_pts).reshape(-1, 1)

  data = numpy.hstack((X1, X2, X3, X4, X5, Y))
  fdf = pd.DataFrame(
      data,
      columns=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'])

  corr = fdf.corr()  # create map
  plt.figure(figsize = (12,10))
  sns.heatmap(round(corr, 3), annot = True,
              vmin=-1, vmax=1, cmap="YlGnBu", linewidths=.5)
  plt.grid(b=True, color='#f68c1f', alpha=0.1)
  plt.show()

def calcC(nt):
    
    c0 = numpy.ones( nt)
    a = numpy.ones( nt)
    dffu= 260
    dt = 0.125 * dx  *dx  / dffu
    #a = dx*dx + dy*dy +dz*dz 
    M =150
    c0 = M / (3/4 * math.pi * a*a*a )
    c = numpy.ones((nt, nt))

    x = numpy.linspace(0, 4, nt)

    for i in range(0,nt):  
                a[i] =i

    for i in range(0,nt):  
                c[i] = c0 * ( 1- math.exp ((-a[i]*a[i])/ (4* dffu*dt)) )
               
 
    fig,ax = pyplot.subplots()
    ax.plot(x,c)   
 
    ax.set_xlabel('$r/a$')
    ax.set_ylabel('$c/c0$')
 

def drwadiffuion():
 
  epsil1=1.0
  epsil2=1.0 
  L=500.0
  Nhalf=25.0 
  T=1000.0
  Ntime=10000 #10000
  dx=L/ (2*Nhalf) 
  dt=T/ (Ntime)  	
  N=2*Nhalf+1 

  amax=260.0;
  amaxgef =1.0

  xi=1.0;
  fc=dt/(xi*dx*dx) 
 
  #definiere Gefäße
  rv=40.0 # in Mikrometer
  rd=50.0  #Radius plus Dicke

  drive=0.0
  epsil1=1.0   #Porosität
  epsil2=1.0   #Porosität
  L=500.0      #die Gefäße des Simulationsraumes in Mikrometer
  Nhalf=25   #Anzahl von Grid-Punkten um das Zentrum für jede Richtung an
  T=1000.0     #die gesamte Simulationszeit
  Ntime=120  # die anzahl von den Zeitschritten an, Ntime
  dx=L/(2*Nhalf) #micro meters
  dt=T/(Ntime) 	
  N=2*Nhalf+1   #gesamte Anzahl von Grid-Punkten
  #definiere Faktor aus Gleichung (3)
  xi=1.0
  fc=dt/(xi*dx*dx)
  # definiere Gefäße
  rv=4   # Radius von Gefäßen in Mikrometer
  rd=5  # in Mikromater Radius plus Wanddick
  kv=0
  kd=0
  r30=1e20
  amaxgef =1.0  #Diffusionskoeffizient in der GefÃ¤ÃŸdicke
  amax=260    #Diffusionskoeffizient
  dt_crit=dx*dx/(8.0*amax) # Stabilitätskriterium
  #Radien fuer Permeabilitaetsberechnung
  Rc=rv+1.0
  Rd=rd+1.0
  u0 = numpy.zeros((N+1,N+1,N+1))
  u = numpy.zeros((N+1,N+1,N+1)) 

  counti =numpy.ones((N+1))
  countj=numpy.ones((N+1))
  countg=numpy.ones((N+1))
  countkdi=numpy.ones((N+1))
  countkdj=numpy.ones((N+1))
  countkdg=numpy.ones((N+1))
  counti[0]=0
  countj[0]=0
  countg[0]=0
  kv=0 ;
  kd=0 ;
  i30 = 0;
  j30 =0;
  g30  = 0;
  for i in range(1,N): 
    for j in range(1,N): 
      for g in range(1,N): 
        r=math.sqrt(((i-Nhalf-1)*(i-Nhalf-1)+(j-Nhalf-1)*(j-Nhalf-1)+(g-Nhalf-1)*(g-Nhalf-1)))*dx;
        if(r<rv):
          kv=kv+1 
          counti[kv]=i 
          countj[kv]=j 
          countg[kv]=g 
 
        if ((r>=rv) &  (r<rd)): 
          kd=kd+1;
          countkdi[kd]=i 
          countkdj[kd]=j 
          countkdg[kd]=g 
 
        if(abs(r-30.0)<r30):   
          r30=r-30.0 
          i30=i 
          j30=j 
          g30 =g
 
 
  dt_crit=dx*dx/(4.0*amax)
  if(dt<dt_crit): #(dt>dt_crit):
    exit()
  else:

    # Zeit-Schleife
    for it in range(1,Ntime):  
     #speichern aif auf aif0 und u auf u0    
     u0 = u.copy()   
     #aif_0 = aif.copy() 
     # or  this
     #for l in range(1,N):  
     #  for m in range(1,N):  
     #     for k in range(1,N): 
     #       u0[l][m][k]=u[l][m][k];       
     #for l in range(1,kv):  
     #  aif_0[l]=aif[l]
       
     #hier wird beachtet, dass die Menge die weggeht von einem Ende des Quadrats, tritt wieder aus anderem Ende ein */
     # X-Schleife
     for i in range(1,N):  
       if(i==N):
        ip=1                    #ip=x+h
       else:
        ip=i+1
        
       if(i==1):
        im=N                    # im=x-h
       else:
        im=i-1
        
     # Y-Schleife
       for j in range(1,N): 
         if(j==N):
           jp=1        
         else:
          jp=j+1
       
         if(j==1):
           jm=N        
         else:
           jm=j-1
       
        
       # hier wird beachtet, dass die Menge die weggeht von einem Ende des Quadrats, tritt wieder aus anderem Ende ein */
       # Z-Schleife
         for g in range(1,N): 
           if(g==N):
             gp=1                    # gp=z+h        
           else:
             gp=g+1        
           if(g==1):
            gm=N                     # gm=z-h 
           else:
            gm=g-1


           #ax_minus=0.5*(a[i][j][g]+a[im][j][g]);
           #ax_plus=0.5*(a[i][j][g]+a[ip][j][g]);
           #ay_minus=0.5*(a[i][j][g]+a[i][jm][g]);
	         #ay_plus=0.5*(a[i][j][g]+a[i][jp][g]);
  	       #az_minus=0.5*(a[i][j][g]+a[i][j][gm]);
  	       #az_plus=0.5*(a[i][j][g]+a[i][j][gp]); 

           # die Funktion
           # u[i][j][g]=u0[i][j][g]+fc*((0.5*(a[i][j][g]+a[im][j][g]))*u0[im][j][g]+(0.5*(a[i][j][g]+a[ip][j][g]))*u0[ip][j][g]+(0.5*(a[i][j][g]+a[i][jm][g]))*u0[i][jm][g]+(0.5*(a[i][j][g]+a[i][jp][g]))*u0[i][jp][g]+(0.5*(a[i][j][g]+a[i][j][gm]))*u0[i][j][gm]+(0.5*(a[i][j][g]+a[i][j][gp]))*u0[ip][j][gp]-((0.5*(a[i][j][g]+a[im][j][g]))+(0.5*(a[i][j][g]+a[ip][j][g]))+(0.5*(a[i][j][g]+a[i][jm][g]))+(0.5*(a[i][j][g]+a[i][jp][g])+(0.5*(a[i][j][g]+a[i][j][gm]))+(0.5*(a[i][j][g]+a[i][j][gp])))*u0[i][j][g]);              
           u[i][j][g]=u0[i][j][g]+((fc*amax*epsil2)*(u0[im][j][g]+u0[ip][j][g]+u0[i][jm][g]+u0[i][jp][g]+u0[i][j][gm]+u0[i][j][gp]-(4.0)*u0[i][j][g])); 
        	 #für die Gefäßdicke , also definiere permeabilität

           #end y-Schleife
           # end x-Schleife
           # end z-Schleife    

    for  p in range(1,kd): 
     u[countkdi[p]][countkdj[p]][countkdg[p]]= u0[countkdi[p]][countkdj[p]][countkdg[p]]+((fc*amaxgef*epsil2)*(u0[countkdi[p]-1][countkdj[p]][countkdg[p]]+u0[countkdi[p]+1][countkdj[p]][countkdg[p]]+u0[countkdi[p]][countkdj[p]-1][countkdg[p]]+u0[countkdi[p]][countkdj[p]+1][countkdg[p]]+u0[countkdi[p]][countkdj[p]][countkdg[p]-1]+u0[countkdi[p]][countkdj[p]][countkdg[p]+1]-(4.0)*u0[countkdi[p]][countkdj[p]][countkdg[p]]));
 
   
    x4 = numpy.linspace(1,N+1,N+1)   
    y4 = numpy.linspace(1,N+1,N+1)
    z4 = numpy.linspace(1,N+1,N+1)
 
    #Plotting
    fig = pyplot.figure()
    X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)
    ax = fig.add_subplot(222, projection='3d')
    img = ax.scatter(X4, Y4, Z4, c=u, cmap=pyplot.jet())
    fig.colorbar(img)

    #for p in range(kd):
    # u[countkdi[p]][countkdj[p]][countkdg[p]]=u0[countkdi[p]][countkdj[p]][countkdg[p]]+((fc*amaxgef*epsil2)*(u0[countkdi[p]-1][countkdj[p]][countkdg[p]]+u0[countkdi[p]+1][countkdj[p]][countkdg[p]]+u0[countkdi[p]][countkdj[p]-1][countkdg[p]]+u0[countkdi[p]][countkdj[p]+1][countkdg[p]]+u0[countkdi[p]][countkdj[p]][countkdg[p]-1]+u0[countkdi[p]][countkdj[p]][countkdg[p]+1]-(4.0)*u0[countkdi[p]][countkdj[p]][countkdg[p]]))
 
def drawmatrix():
  fig  = pyplot.subplots()
  X1 = numpy.array([100,200,300,400, 500])
  X2 = numpy.array([1000,1500,2000,1800, 3000])
  X3 = numpy.array([2,2,3,4, 4])
  X4 = numpy.array([50,100,200,0, 50])
  fig = pyplot.figure()
  ax= fig.add_subplot(1,1,1, projection='3d') #add_subplot(111, projection='3d') same ,for postion
  ax.scatter3D(X1,X2,X3, c=X4)
  pyplot.show()


def MPI_temp():
    """
    convergence =0
    while (!convergence)    
        /* Schrittweite und Zeit inkrementieren*/
        step = step + 1;
        t = t + dt;

        /* FÃ¼hren  einen Schritt des expliziten Schemas aus */
        computeNext(x0, x, size_tot_x, size_tot_y, size_tot_z, dt, hx, hy, hz, &resLoc, me, xs, ys, zs, xe, ye, ze, k0);

        /*  Aktualisieren die TeillÃ¶sung entlang der Schnittstelle */
        updateBound(x0, size_tot_x, size_tot_y, size_tot_z, NeighBor, comm3d,
            matrix_type_oxz, matrix_type_oxy, matrix_type_oyz, me, xs, ys, zs, xe, ye, ze);

        /* Summenverminderung, um Fehler zu erhalten*/
        MPI_Allreduce(&resLoc, &result, 1, MPI_DOUBLE, MPI_SUM, comm);

        /* Aktueller Fehler */
        result = sqrt(result);

        /* Bruchbedingungen von main loop */
        if ((result < epsilon) || (step > maxStep)) break;
    """


def MPI4py():
  rank =-1;
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  if rank == 0:
    print("First rank")
  elif rank == 1:
    print ("Second rank")
  else:
    print("Not first or second rank")
    
  print("Hello world from rank", str(rank), "of", str(size))  


def my_MPi():
 
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()



  params = numpy.random.random((15, 3)) * 100.0  # parameters to send to my_function
  n = params.shape[0]

  count = n // size  # number of catchments for each process to analyze
  remainder = n % size  # extra catchments if n is not a multiple of size

  if rank < remainder:  # processes with rank < remainder analyze one extra catchment
      start = rank * (count + 1)  # index of first catchment to analyze
      stop = start + count + 1  # index of last catchment to analyze
  else:
      start = rank * count + remainder
      stop = start + count

  local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank
  local_results = numpy.empty((local_params.shape[0], local_params.shape[1] + 1))  # create result array
  local_results[:, :local_params.shape[1]] = local_params  # write parameter values to result array
  local_results[:, -1] = my_function(local_results[:, 0], local_results[:, 1], local_results[:, 2])  # run the function for each parameter set and rank

  # send results to rank 0
  rank_size = -1
  if rank > 0:
      comm.Send(local_results, dest=0, tag=14)  # send results to process 0
  else:
      final_results = numpy.copy(local_results)  # initialize final results with results from process 0
      for i in range(1, size):  # determine the size of the array to be received from each process
          if i < remainder:
              rank_size = count + 1
          else:
              rank_size = count
          tmp = numpy.empty((rank_size, final_results.shape[1]), dtype=numpy.float)  # create empty array to receive results
          comm.Recv(tmp, source=i, tag=14)  # receive results from the process
          final_results = numpy.vstack((final_results, tmp))  # add the received results to the final results
      print("results" )
      print(final_results , " size = " ,size , " rank " , rank , "  rank_size " , rank_size)

def my_function1(param1, param2 , param3 ):
    result = param1 ** 2 * param2 + param3 
    return result

def my_function(param1, param2 , param3 ):
    result = param1 + param2 
    time.sleep(2)
    return result 
 
def my_MPi1():
 
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()



  params = numpy.random.random((4, 3)) * 100.0  # parameters to send to my_function
  n = params.shape[0]

  count = n // size  # number of catchments for each process to analyze
  remainder = n % size  # extra catchments if n is not a multiple of size

  if rank < remainder:  # processes with rank < remainder analyze one extra catchment
      start = rank * (count + 1)  # index of first catchment to analyze
      stop = start + count + 1  # index of last catchment to analyze
  else:
      start = rank * count + remainder
      stop = start + count

  local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank
  local_results = numpy.empty((local_params.shape[0], local_params.shape[1] + 1))  # create result array
  local_results[:, :local_params.shape[1]] = local_params  # write parameter values to result array
  
  local_results[:, :] = my_function(local_results[:, 0], local_results[:, 1], local_results[:, 2])  # run the function for each parameter set and rank
  print("local_results" )
  print(local_results  )
  # send results to rank 0
  rank_size = -1
  if rank > 0:
      comm.Send(local_results, dest=0, tag=14)  # send results to process 0
  else:
      final_results = numpy.copy(local_results)  # initialize final results with results from process 0
      for i in range(1, size):  # determine the size of the array to be received from each process
          if i < remainder:
              rank_size = count + 1
          else:
              rank_size = count
          tmp = numpy.empty((rank_size, final_results.shape[1]), dtype=numpy.float)  # create empty array to receive results
          comm.Recv(tmp, source=i, tag=14)  # receive results from the process
          final_results = numpy.vstack((final_results, tmp))  # add the received results to the final results
      print("results" )
      print(final_results , " size = " ,size , " rank " , rank , "  rank_size " , rank_size)

def newarray():
  comm = MPI.COMM_WORLD
  rank = comm.rank
  size = comm.size

  if rank >= size/2:
      nb_elts = 5
  else:
      nb_elts = 2

  # create data
  lst = []
  for i in range(nb_elts):
      lst.append(rank*3+i)
  array_lst = numpy.array(lst, dtype=int)

  # communicate array
  result = []
  if rank == 0:
      result = array_lst
      for p in range(1, size):

          if p >= size/2:
               nb_elts = 5
          else:
               nb_elts = 2

          received = numpy.empty(nb_elts, dtype=int)
          comm.Recv(received, p, tag=13)
          result = numpy.concatenate([result, received])
  else:
      comm.Send(array_lst, 0, tag=13)

  if rank==0:
      print ( "Send Recv, result= " , str(result))

  #How to use Gatherv:
  nbsum=0
  sendcounts=[]
  displacements=[]

  for p in range(0,size):
      displacements.append(nbsum)
      if p >= size/2:
               nbsum+= 5
               sendcounts.append(5)
      else:
               nbsum+= 2
               sendcounts.append(2)

  if rank==0:
      print ("nbsum "+str(nbsum))
      print ("sendcounts "+str(tuple(sendcounts)))
      print ("displacements "+str(tuple(displacements)))
  print ("rank "+str(rank)+" array_lst "+str(array_lst))
  print ("numpy.int "+str(numpy.dtype(int))+" "+str(numpy.dtype(int).itemsize)+" "+str(numpy.dtype(int).name) )

  if rank==0:
      result2=numpy.empty(nbsum, dtype=int)
  else:
      result2=None

  comm.Gatherv(array_lst,[result2,tuple(sendcounts),tuple(displacements),MPI.DOUBLE],root=0)

  if rank==0:
      print ("Gatherv, result2= "+str(result2))
def mpigood2():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  print("size =", size)
  print("rank=", rank)
  a = 1
  b = 1000

  num_per_rank = b // size # the floor division // rounds the result down to the nearest whole number.
  summ = numpy.zeros(1)

  temp = 0
  lower_bound = a + rank * num_per_rank
  upper_bound = a + (rank + 1) * num_per_rank
  print("This is processor ", rank, "and I am summing numbers from", lower_bound," to ", upper_bound - 1, flush=True)

  comm.Barrier()
  start_time = time.time()

  for i in range(lower_bound, upper_bound):
      temp = temp + i

  summ[0] = temp

  if rank == 0:
      total = numpy.zeros(1)
  else:
      total = None

  comm.Barrier()
  # collect the partial results and add to the total sum
  comm.Reduce(summ, total, op=MPI.SUM, root=0)

  stop_time = time.time()

  if rank == 0:
    # add the rest numbers to 1 000 000
    for i in range(a + (size) * num_per_rank, b + 1):
        total[0] = total[0] + i
    print("The sum of numbers from 1 to 1 000 000: ", int(total[0]))
    print("time spent with ", size, " threads in milliseconds")
    print("-----", int((time.time() - start_time) * 1000), "-----")

def mains():
    comm = MPI.COMM_WORLD
    id = comm.Get_rank()            #number of the process running the code
    numProcesses = comm.Get_size()  #total number of processes running
    myHostName = MPI.Get_processor_name()  #machine name running the code
    params = numpy.random.random((8, 2)) * 4.0 
    REPS = 8

    if (numProcesses <= REPS):

        for i in range(id, REPS, numProcesses):
            print("On {}: Process {} is performing iteration {}"\
            .format(myHostName, id, i))

    else:
        # can't hove more processes than work; one process reports the error
        if id == 0 :
            print("Please run with number of processes less than \or equal to {}.".format(REPS))

def mpigood():
  # get number of processors and processor rank
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  params = numpy.random.random((8, 2)) * 4.0  # parameters to send to my_function
  n = params.shape[0]

  count = n // size  # number of catchments for each process to analyze
  remainder = n % size  # extra catchments if n is not a multiple of size

  if rank < remainder:  # processes with rank < remainder analyze one extra catchment
      start = rank * (count + 1)  # index of first catchment to analyze
      stop = start + count + 1  # index of last catchment to analyze
  else:
      start = rank * count + remainder
      stop = start + count

  local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank
  local_results = numpy.empty((local_params.shape[0], local_params.shape[1] + 1))  # create result array
  local_results[:, :local_params.shape[1]] = local_params  # write parameter values to result array
  local_results[:, -1] = my_function(local_results[:, 0], local_results[:, 1], local_results[:, 2])  # run the function for each parameter set and rank

  print("start:  = " ,start , "  ")
  print("stop  = " ,stop , "  ")
  # send results to rank 0
  if rank > 0:
      comm.Send(local_results, dest=0, tag=14)  # send results to process 0
      print("local_results  = " ,0 , "  ")
  else:
      final_results = numpy.copy(local_results)  # initialize final results with results from process 0
      for i in range(1, size):  # determine the size of the array to be received from each process
          print("i  = " ,i , "  ")
          if i < remainder:
              rank_size = count + 1
          else:
              rank_size = count
          tmp = numpy.empty((rank_size, final_results.shape[1]), dtype=float)  # create empty array to receive results
          #tmp = numpy.empty((rank_size, final_results.shape[1]), dtype=numpy.float)  # create empty array to receive results
          comm.Recv(tmp, source=i, tag=14)  # receive results from the process
          final_results = numpy.vstack((final_results, tmp))  # add the received results to the final results
          print("rank_size  = " ,rank_size , "  ")
      print("rank = " , rank , "  ")
      print(final_results)


def calc_Sum_u(it,dx,N, u,c_ana,  Sum_u,Sum_ana ):
     Sum_u[it] = 0;
     Sum_ana[it] = 0;
     for i in range(1,N): 
       for j in range(1,N): 
          for k in range(1,N): 
              Sum_u[it] = Sum_u[it] + u[i][j][k];
              Sum_ana[it] = Sum_ana[it] + c_ana[i][j][k];
     
     Sum_u[it] = Sum_u[it] * dx  * dx * (6.022e23); #Sum_u[it] * dx * dx * (6.022e23);
     Sum_ana[it] = Sum_ana[it] *  dx * dx * (6.022e23); #Sum_ana[it] * dx * dx * (6.022e23);

    
def calc_function(it,N,u0,u,c_ana,dt,  fc,Nhalf,dx,Dif,M_PI,  amax ,epsil2 ):
    
     u0 = u.copy()
     #aif_0 = aif.copy() 
     # or  this
     #for l in range(1,N):  
     #  for m in range(1,N):  
     #     for k in range(1,N): 
     #       u0[l][m][k]=u[l][m][k];       
     #for l in range(1,kv):  
     #  aif_0[l]=aif[l]
       
     #hier wird beachtet, dass die Menge die weggeht von einem Ende des Quadrats, tritt wieder aus anderem Ende ein */
     # X-Schleife
     for i in range(1,N):  
       if(i==N):
        ip=1                    #ip=x+h
       else:
        ip=i+1
        
       if(i==1):
        im=N                    # im=x-h
       else:
        im=i-1
        
     # Y-Schleife
       for j in range(1,N): 
         if(j==N):
           jp=1        
         else:
          jp=j+1
       
         if(j==1):
           jm=N        
         else:
           jm=j-1
       
        
       # hier wird beachtet, dass die Menge die weggeht von einem Ende des Quadrats, tritt wieder aus anderem Ende ein */
       # Z-Schleife
         for g in range(1,N): 
           if(g==N):
             gp=1                    # gp=z+h        
           else:
             gp=g+1        
           if(g==1):
            gm=N                     # gm=z-h 
           else:
            gm=g-1


           #ax_minus=0.5*(a[i][j][g]+a[im][j][g]);
           #ax_plus=0.5*(a[i][j][g]+a[ip][j][g]);
           #ay_minus=0.5*(a[i][j][g]+a[i][jm][g]);
	         #ay_plus=0.5*(a[i][j][g]+a[i][jp][g]);
  	       #az_minus=0.5*(a[i][j][g]+a[i][j][gm]);
  	       #az_plus=0.5*(a[i][j][g]+a[i][j][gp]); 

           # die Funktion
           # u[i][j][g]=u0[i][j][g]+fc*((0.5*(a[i][j][g]+a[im][j][g]))*u0[im][j][g]+(0.5*(a[i][j][g]+a[ip][j][g]))*u0[ip][j][g]+(0.5*(a[i][j][g]+a[i][jm][g]))*u0[i][jm][g]+(0.5*(a[i][j][g]+a[i][jp][g]))*u0[i][jp][g]+(0.5*(a[i][j][g]+a[i][j][gm]))*u0[i][j][gm]+(0.5*(a[i][j][g]+a[i][j][gp]))*u0[ip][j][gp]-((0.5*(a[i][j][g]+a[im][j][g]))+(0.5*(a[i][j][g]+a[ip][j][g]))+(0.5*(a[i][j][g]+a[i][jm][g]))+(0.5*(a[i][j][g]+a[i][jp][g])+(0.5*(a[i][j][g]+a[i][j][gm]))+(0.5*(a[i][j][g]+a[i][j][gp])))*u0[i][j][g]);              
           u[i][j][g]=5#u0[i][j][g]+((fc*amax*epsil2)*(u0[im][j][g]+u0[ip][j][g]+u0[i][jm][g]+u0[i][jp][g]+u0[i][j][gm]+u0[i][j][gp]-(4.0)*u0[i][j][g])); 
        	 #für die Gefäßdicke , also definiere permeabilität

           tim = dt * it ;

           c_ana[i][j][g] =2#(16.60577881 / (4.0 * M_PI * Dif * tim)) * math.exp(-(dx * dx * ( ((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))) / (4.0 * Dif * (tim))));

     return u

def drwadiffuionMPI():
 
  epsil1=1.0
  epsil2=1.0 
  L=500.0
  Nhalf=3 #25.0 
  T=1000.0
  Ntime=10000 #10000
  dx=L/ (2*Nhalf) 
  dt=T/ (Ntime)  	
  N=2*Nhalf+1 

  amax=260.0;
  amaxgef =1.0

  xi=1.0;
  fc=dt/(xi*dx*dx) 
 
  #definiere Gefäße
  rv=40.0 # in Mikrometer
  rd=50.0  #Radius plus Dicke

  drive=0.0
  epsil1=1.0   #Porosität
  epsil2=1.0   #Porosität
  L=500.0      #die Gefäße des Simulationsraumes in Mikrometer
  Nhalf=25   #Anzahl von Grid-Punkten um das Zentrum für jede Richtung an
  T=1000.0     #die gesamte Simulationszeit
  Ntime=8  # die anzahl von den Zeitschritten an, Ntime
  dx=L/(2*Nhalf) #micro meters
  dt=T/(Ntime) 	
  N=2*Nhalf+1   #gesamte Anzahl von Grid-Punkten
  #definiere Faktor aus Gleichung (3)
  xi=1.0
  fc=dt/(xi*dx*dx)
  # definiere Gefäße
  rv=4   # Radius von Gefäßen in Mikrometer
  rd=5  # in Mikromater Radius plus Wanddick
  kv=0
  kd=0
  r30=1e20
  amaxgef =1.0  #Diffusionskoeffizient in der GefÃ¤ÃŸdicke
  amax=260    #Diffusionskoeffizient
  dt_crit=dx*dx/(8.0*amax) # Stabilitätskriterium
  #Radien fuer Permeabilitaetsberechnung
  Rc=rv+1.0
  Rd=rd+1.0
  u0 = numpy.zeros((N+1,N+1))
  u = numpy.zeros((N+1 ,N+1)) 
  c_ana = numpy.zeros((N+1 ,N+1)) 

  Sum_u    =numpy.zeros((Ntime)) 
  Sum_ana = numpy.zeros((Ntime)) 
  M_PI =math.pi;
  counti =numpy.ones((N+1))
  countj=numpy.ones((N+1))
  countg=numpy.ones((N+1))
  countkdi=numpy.ones((N+1))
  countkdj=numpy.ones((N+1))
  countkdg=numpy.ones((N+1))
  counti[0]=0
  countj[0]=0
  countg[0]=0
  kv=0 ;
  kd=0 ;
  i30 = 0;
  j30 =0;
  g30  = 0;
  Dif = 260.0;
  M = 16.60577881 / (dx * dx * dx);  #16.60577881 / (dx * dx);  
  for i in range(1,N): 
    for j in range(1,N): 
      for g in range(1,N): 
        r=math.sqrt(((i-Nhalf-1)*(i-Nhalf-1)+(j-Nhalf-1)*(j-Nhalf-1)+(g-Nhalf-1)*(g-Nhalf-1)))*dx;
        if(r<rv):
          kv=kv+1 
          counti[kv]=i 
          countj[kv]=j 
          countg[kv]=g 
 
        if ((r>=rv) &  (r<rd)): 
          kd=kd+1;
          countkdi[kd]=i 
          countkdj[kd]=j 
          countkdg[kd]=g 
 
        if(abs(r-30.0)<r30):   
          r30=r-30.0 
          i30=i 
          j30=j 
          g30 =g
 
 
  dt_crit=dx*dx/(4.0*amax)
  if(dt<dt_crit): #(dt>dt_crit):
    exit()
  else:

    # Zeit-Schleife # here we start MPI steps
    Sum_u =  numpy.zeros((Ntime))
    Sum_u1 =numpy.zeros((Ntime))
    Sum_ana = numpy.zeros((Ntime))
    Sum_ana1 = numpy.zeros((Ntime))
    result = []
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    params = numpy.ones((N+1, N+1,N+1))    # parameters to send to calc_function
    n = params.shape[0]

    count = n // size  # number of catchments for each process to analyze
    remainder = n % size  # extra catchments if n is not a multiple of size

 
    tag_data = 42
    tag_end = 23
    a = 0
    b = Ntime
    num_per_rank = b  // size
 
    lower_bound = a + rank * num_per_rank
    upper_bound = a + (rank + 1) * num_per_rank
   
    start_time = time.time()


    for it in range(lower_bound, upper_bound):
     local_resultstemp  = calc_function(it,N,u0,u,c_ana, dt,  fc, Nhalf,dx,Dif,M_PI, amax ,epsil2    )  # run the function for each parameter set and rank
     calc_Sum_u(it,dx,N, u,c_ana,  Sum_u,Sum_ana )
     
    #outputData = comm.gather(Sum_u,root=0)
     #print("This is processor ", rank, "and I am summing numbers from", lower_bound," to ", upper_bound - 1, flush=True)


 
    
    #comm.Reduce(local_results, local_results1, op=MPI.SUM, root=0)
    # collect the partial results and add to the total sum
    #comm.Reduce(summ, total, op=MPI.SUM, root=0)
    stop_time = time.time() 
    if rank == 0:
      c_ana1 = c_ana
      uu1 = u
      Sum_u1 =Sum_u.copy()
      Sum_ana1= Sum_ana.copy()
      for p in range(1, size): 
          received = numpy.empty(Ntime, dtype=float)
          received1 = numpy.empty(Ntime, dtype=float)  
          rank_size=  numpy.empty(1, dtype=int)
          #received2= numpy.empty(Ntime, dtype=float)
          comm.Recv(received, p, tag=tag_data)
          comm.Recv(received1, p, tag=tag_data)
          comm.Recv(c_ana1, p, tag=tag_data)
          comm.Recv(uu1, p, tag=tag_data)
          comm.Recv(rank_size, p, tag=tag_data)
          lower_bound = a+  rank_size[0] * num_per_rank
          upper_bound = a+ (rank_size[0] + 1) * num_per_rank
          #c_ana1 = received2 
          for k in  range(lower_bound, upper_bound):
             
               Sum_u1[k]= received[k]
               Sum_ana1[k]= received1[k] 
              
    else:
      lower_bound1 = numpy.empty(1, dtype=int)
      lower_bound1[0] = rank
      comm.Send(Sum_u, dest=0, tag=tag_data)
      comm.Send(Sum_ana, dest=0, tag=tag_data)
      comm.Send(c_ana, dest=0, tag=tag_data)      
      comm.Send(u, dest=0, tag=tag_data)
      comm.Send(lower_bound1, dest=0, tag=tag_data)
      comm.send(None, dest=0, tag=tag_end)
 
    #comm.Barrier()

    #rank_size = -1
    #if rank > 0:
    #  comm.Send(  Sum_u, dest=0, tag=14)  # send results to process 0
    if rank == 0:
      print(" Sum_u = " , Sum_u1)
      print(" Sum_ana = " , Sum_ana1)
      print(" u = " , u)      
      print(" c_ana = " , c_ana1)
      #print(" uu1 = " , uu1)
      

    #  print("Sum_u= " ,Sum_u )
 
    #u=final_results.copy()

    #for  p in range(1,kd): 
    # u[countkdi[p]][countkdj[p]][countkdg[p]]= u0[countkdi[p]][countkdj[p]][countkdg[p]]+((fc*amaxgef*epsil2)*(u0[countkdi[p]-1][countkdj[p]][countkdg[p]]+u0[countkdi[p]+1][countkdj[p]][countkdg[p]]+u0[countkdi[p]][countkdj[p]-1][countkdg[p]]+u0[countkdi[p]][countkdj[p]+1][countkdg[p]]+u0[countkdi[p]][countkdj[p]][countkdg[p]-1]+u0[countkdi[p]][countkdj[p]][countkdg[p]+1]-(4.0)*u0[countkdi[p]][countkdj[p]][countkdg[p]]));
 

    x4 = numpy.linspace(1,N+1,N+1)   
    y4 = numpy.linspace(1,N+1,N+1)
    z4 = numpy.linspace(1,N+1,N+1)
 
    #Plotting
    fig = pyplot.figure()
    X4, Y4, Z4 = numpy.meshgrid(x4, y4,z4)
    ax = fig.add_subplot(222, projection='3d')
    img = ax.scatter(X4, Y4, Z4, c=u, cmap=pyplot.jet())
    fig.colorbar(img)



def calc_Sum_u2D(it,dx,N, u,c_ana,  Sum_u,Sum_ana ):
     Sum_u[it] = 0;
     Sum_ana[it] = 0;
     for i in range(1,N): 
       for j in range(1,N): 

              Sum_u[it] = Sum_u[it] + u[i][j] ;
              Sum_ana[it] = Sum_ana[it] + c_ana[i][j];
     
     Sum_u[it] = Sum_u[it] * dx  * dx * (6.022e23); #Sum_u[it] * dx * dx * (6.022e23);
     Sum_ana[it] = Sum_ana[it] *  dx * dx * (6.022e23); #Sum_ana[it] * dx * dx * (6.022e23);

    
def calc_function2D(it,N,u0,u,c_ana,dt,  fc,Nhalf,dx,Dif,M_PI,  amax ,epsil2 ):
    
     u0 = u.copy()
     #aif_0 = aif.copy() 
     # or  this
     #for l in range(1,N):  
     #  for m in range(1,N):  
     #     for k in range(1,N): 
     #       u0[l][m][k]=u[l][m][k];       
     #for l in range(1,kv):  
     #  aif_0[l]=aif[l]
       
     #hier wird beachtet, dass die Menge die weggeht von einem Ende des Quadrats, tritt wieder aus anderem Ende ein */
     # X-Schleife
     for i in range(1,N):  
       if(i==N):
        ip=1                    #ip=x+h
       else:
        ip=i+1
        
       if(i==1):
        im=N                    # im=x-h
       else:
        im=i-1
        
     # Y-Schleife
       for j in range(1,N): 
         if(j==N):
           jp=1        
         else:
          jp=j+1
       
         if(j==1):
           jm=N        
         else:
           jm=j-1                
           #ax_minus=0.5*(a[i][j][g]+a[im][j][g]);
           #ax_plus=0.5*(a[i][j][g]+a[ip][j][g]);
           #ay_minus=0.5*(a[i][j][g]+a[i][jm][g]);
	         #ay_plus=0.5*(a[i][j][g]+a[i][jp][g]);
  	       #az_minus=0.5*(a[i][j][g]+a[i][j][gm]);
  	       #az_plus=0.5*(a[i][j][g]+a[i][j][gp]); 

           # die Funktion
           # u[i][j][g]=u0[i][j][g]+fc*((0.5*(a[i][j][g]+a[im][j][g]))*u0[im][j][g]+(0.5*(a[i][j][g]+a[ip][j][g]))*u0[ip][j][g]+(0.5*(a[i][j][g]+a[i][jm][g]))*u0[i][jm][g]+(0.5*(a[i][j][g]+a[i][jp][g]))*u0[i][jp][g]+(0.5*(a[i][j][g]+a[i][j][gm]))*u0[i][j][gm]+(0.5*(a[i][j][g]+a[i][j][gp]))*u0[ip][j][gp]-((0.5*(a[i][j][g]+a[im][j][g]))+(0.5*(a[i][j][g]+a[ip][j][g]))+(0.5*(a[i][j][g]+a[i][jm][g]))+(0.5*(a[i][j][g]+a[i][jp][g])+(0.5*(a[i][j][g]+a[i][j][gm]))+(0.5*(a[i][j][g]+a[i][j][gp])))*u0[i][j][g]);              
         
         #u[i][j]=i+j # u0[i][j] +((fc*amax*epsil2)*(u0[im][j] +u0[ip][j] +u0[i][jm] +u0[i][jp] +u0[i][j] +u0[i][j] -(4.0)*u0[i][j] )); 
         tt = u0[i][j] + (((dt * Dif) / (dx * dx)) * (u0[im][j] + u0[ip][j] + u0[i][jm] + u0[i][jp] - 4.0 * u0[i][j]));
         #if (tt <0):
         #  tt= tt* -1
         u[i][j] =   tt 
       	 #für die Gefäßdicke , also definiere permeabilität
         tim = dt * it ;  
         if (tim == 0):
           tim = 1

           #c_ana[i][j][g] =2#(16.60577881 / (4.0 * M_PI * Dif * tim)) * math.exp(-(dx * dx * ( ((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))) / (4.0 * Dif * (tim))));
         c_ana[i][j] = (16.60577881 / (4.0 * M_PI * Dif * tim)) *  math.exp(-(dx * dx * (((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))) / (4.0 * Dif * (tim))));
     
     #return u
 

def drwadiffuionMPI2D():
 
  epsil1=1.0
  epsil2=1.0 
  L=500.0
  Nhalf=3 #25.0 
  T=1000.0
  Ntime=50000 #10000
  dx=L/ (2*Nhalf) 
  dt=T/ (Ntime)  	
  N=2*Nhalf+1 

  amax=260.0;
  amaxgef =1.0

  xi=1.0;
  fc=dt/(xi*dx*dx) 
 
  #definiere Gefäße
  rv=40.0 # in Mikrometer
  rd=50.0  #Radius plus Dicke

  drive=0.0
  epsil1=1.0   #Porosität
  epsil2=1.0   #Porosität
  L=500.0      #die Gefäße des Simulationsraumes in Mikrometer
  Nhalf=50   #Anzahl von Grid-Punkten um das Zentrum für jede Richtung an
  T=1000.0     #die gesamte Simulationszeit
  Ntime=100000  # die anzahl von den Zeitschritten an, Ntime
  dx=L/(2*Nhalf) #micro meters
  dt=T/(Ntime) 	
  Ntime=400
  N=2*Nhalf+1   #gesamte Anzahl von Grid-Punkten
  #definiere Faktor aus Gleichung (3)
  xi=1.0
  fc=dt/(xi*dx*dx)
  # definiere Gefäße
  rv=4   # Radius von Gefäßen in Mikrometer
  rd=5  # in Mikromater Radius plus Wanddick
  kv=0
  kd=0
  r30=1e20
  amaxgef =1.0  #Diffusionskoeffizient in der GefÃ¤ÃŸdicke
  amax=260    #Diffusionskoeffizient
  dt_crit=dx*dx/(4.0*amax) # Stabilitätskriterium
  #Radien fuer Permeabilitaetsberechnung
  Rc=rv+1.0
  Rd=rd+1.0
  u0 = numpy.zeros((N+1,N+1))  
  u =numpy.ones((N+1,N+1))  # numpy.random.normal(0.00000001,0.0000000000001,(N+1,N+1))
  c_ana = numpy.zeros((N+1,N+1))  #numpy.random.normal(0.00000001,0.00000001,(N+1,N+1))

  Sum_u    =numpy.zeros((Ntime+1)) 
  Sum_ana = numpy.zeros((Ntime+1)) 
  M_PI =math.pi;
  counti =numpy.ones((N+1))
  countj=numpy.ones((N+1))
  countg=numpy.ones((N+1))
  countkdi=numpy.ones((N+1))#numpy.ones((N*N+2)) or #numpy.ones((N*N*N+2))
  countkdj=numpy.ones((N+1))#numpy.ones((N*N+2)) or #numpy.ones((N*N*N+2))
  countkdg=numpy.ones((N+1)) #numpy.ones((N*N+2)) or #numpy.ones((N*N*N+2))
  counti[0]=0
  countj[0]=0
 
  N_start = 0;
  Ntime_start = 0;
  N_end = 0;
  Ntime_end = 0;
  kv=0 ;
  kd=0 ;
  i30 = 0;
  j30 =0;
 
  Dif = 260.0;
  M = 16.60577881 / ( dx * dx);  #16.60577881 / (dx * dx);  
  for i in range(0,N+1): 
    for j in range(0,N+1): 
        #u[i][j] = 0.000000000001+random.randint(1,1000); 
        #u[i][j] = 0.000000000001+random.randint(1,1000); 
        u[i][j] = 0.000000001 +0.000000001 *(i+j)  ; 
 
 

  for i in range(1,N): 
    for j in range(1,N): 
      
        r=dx * math.sqrt( ((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))); 
        if(r<rv):
          kv=kv+1 
          counti[kv]=i 
          countj[kv]=j 
 
 
        if ((r>=rv) &  (r<rd)): 
          kd=kd+1;
          countkdi[kd]=i 
          countkdj[kd]=j 
 
 
        if(abs(r-30.0)<r30):   
          r30=r-30.0 
          i30=i 
          j30=j 
 
 
 
  dt_crit=dx*dx/(4.0*amax) #8.0 * Dif 3d
  if(dt>dt_crit): 
    print(" None  " ,dt )
    exit()
  else:
    Ntime =100
    # Zeit-Schleife # here we start MPI steps
    Sum_u =  numpy.zeros((Ntime))
    Sum_u1 =numpy.zeros((Ntime))
    Sum_ana = numpy.zeros((Ntime))
    Sum_ana1 = numpy.zeros((Ntime))
    result = []
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #params = numpy.ones((N+1, N+1,N+1))    # parameters to send to calc_function
    #n = params.shape[0]

    #count = n // size  # number of catchments for each process to analyze
    #remainder = n % size  # extra catchments if n is not a multiple of size

 
    tag_data = 42
    tag_end = 23
    a = 0
    b = Ntime
    num_per_rank = b  // size
 
    lower_bound = a + rank * num_per_rank
    upper_bound = a + (rank + 1) * num_per_rank
   
    start_time = time.time()


    for it in range(lower_bound, upper_bound):
     calc_function2D(it,N,u0,u,c_ana, dt,  fc, Nhalf,dx,Dif,M_PI, amax ,epsil2    )  # run the function for each parameter set and rank
     calc_Sum_u2D(it,dx,N, u,c_ana,  Sum_u,Sum_ana )
     
    #outputData = comm.gather(Sum_u,root=0)
     #print("This is processor ", rank, "and I am summing numbers from", lower_bound," to ", upper_bound - 1, flush=True)


 
    
    #comm.Reduce(local_results, local_results1, op=MPI.SUM, root=0)
    # collect the partial results and add to the total sum
    #comm.Reduce(summ, total, op=MPI.SUM, root=0)
    stop_time = time.time() 
    if rank == 0:
      c_ana1 = c_ana.copy()
      uu1 = u.copy()
      Sum_u1 =Sum_u.copy()
      Sum_ana1= Sum_ana.copy()
      for p in range(1, size): 
          received = numpy.empty(Ntime, dtype=float)
          received1 = numpy.empty(Ntime, dtype=float)  
          rank_size=  numpy.empty(1, dtype=int)
          #received2= numpy.empty(Ntime, dtype=float)
          comm.Recv(received, p, tag=tag_data)
          comm.Recv(received1, p, tag=tag_data)
          comm.Recv(c_ana1, p, tag=tag_data)
          comm.Recv(uu1, p, tag=tag_data)
          comm.Recv(rank_size, p, tag=tag_data)
          lower_bound = a+  rank_size[0] * num_per_rank
          upper_bound = a+ (rank_size[0] + 1) * num_per_rank
          #c_ana1 = received2 
          for k in  range(lower_bound, upper_bound):
             
               Sum_u1[k]= received[k]
               Sum_ana1[k]= received1[k] 
              
    else:
      lower_bound1 = numpy.empty(1, dtype=int)
      lower_bound1[0] = rank
      comm.Send(Sum_u, dest=0, tag=tag_data)
      comm.Send(Sum_ana, dest=0, tag=tag_data)
      comm.Send(c_ana, dest=0, tag=tag_data)      
      comm.Send(u, dest=0, tag=tag_data)
      comm.Send(lower_bound1, dest=0, tag=tag_data)
      comm.send(None, dest=0, tag=tag_end)
 
    #comm.Barrier()

    #rank_size = -1
    #if rank > 0:
    #  comm.Send(  Sum_u, dest=0, tag=14)  # send results to process 0
    # N_start = 0;
    # Ntime_start = 0;
    # N_end = 0;
    #Ntime_end = 0;
    Ntime_start =lower_bound;
    N_start =lower_bound;
    if rank == 0:
      Sum_uf = open("d:/Sum_uf.txt", "w")
      Sum_anaf = open("d:/Sum_anaf.txt", "w")
      uf = open("d:/uf.txt", "w")
      c_anaf = open("d:/c_ana.txt", "w")
      
      print(" Sum_u = " , Sum_u1)
      print(" Sum_ana = " , Sum_ana1)
      print(" u = " , u)      
      print(" c_ana = " , c_ana1)
      for it in range(1,Ntime): 
        Sum_uf.write( str((it + Ntime_start) * dt) +  "    " +str( Sum_u1[it]) +"\n")
        Sum_anaf.write(str((it + Ntime_start) * dt)+ "    " + str (Sum_ana1[it]) +"\n")

      for i in range(1,N): 
          for j in range(1,N): 
                uf.write( str((i + N_start) * dx) +  "    " + str((j + N_start) * dx) +  "    " + str( u[i][j]) +"\n")
                c_anaf.write( str((i + Ntime_start) * dx) +  "    " +  "    " + str((j + N_start) * dx) +  "    " + str( c_ana[i][j]) +"\n")
      Sum_uf.close()
      Sum_anaf.close()
      uf.close()
      c_anaf.close()
    if rank == 0:

      #draw fig 1,good
      #x = numpy.linspace(0, 20000  , N+1)
      #y = numpy.linspace(0, 2000 ,N+1)
      #ax.set_xlim(100, 2000)
      #ax.set_ylim(100, 2000)
      #ax.set_zlim(0.00000001,  0.0000000000000000000001)

      # draw fig 2,not good
      # ax.set_zlim(0.00000001,  0.0000000000000000000001)
      #x = numpy.linspace(1000, 25000  , N+1)
      #y = numpy.linspace(1000, 2500 ,N+1)

      # this good 
      #x = numpy.linspace(0, 10, N+1)
      #y = numpy.linspace(0, 10, N+1)

      x = numpy.linspace(1, N+1, N+1)
      y = numpy.linspace(1, N+1, N+1)

      X, Y = numpy.meshgrid(x, y)

  
      fig = pyplot.figure()
      ax = fig.gca(projection='3d')
      #ax.set_xlim(0, 1500)
      #ax.set_ylim(0, 1500)
      #ax.set_zlim(0.00000000001 , 0.000000001 )
   
      surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, #cmap=cm.viridis,
          linewidth=0, antialiased=True)
 
      ax.set_xlabel('$x$')
      ax.set_ylabel('$y$');

 
      pyplot.show()
      #print(" uu1 = " , uu1)
      
 
 

def drwadiffuionMPI2DnoMPI():
 
  epsil1=1.0
  epsil2=1.0 
  L=500.0
  Nhalf=3 #25.0 
  T=1000.0
  Ntime=50000 #10000
  dx=L/ (2*Nhalf) 
  dt=T/ (Ntime)  	
  N=2*Nhalf+1 

  amax=260.0;
  amaxgef =1.0

  xi=1.0;
  fc=dt/(xi*dx*dx) 
 
  #definiere Gefäße
  rv=40.0 # in Mikrometer
  rd=50.0  #Radius plus Dicke

  drive=0.0
  epsil1=1.0   #Porosität
  epsil2=1.0   #Porosität
  L=500.0      #die Gefäße des Simulationsraumes in Mikrometer
  Nhalf=50   #Anzahl von Grid-Punkten um das Zentrum für jede Richtung an
  T=1000.0     #die gesamte Simulationszeit
  Ntime=100000  # die anzahl von den Zeitschritten an, Ntime
  dx=L/(2*Nhalf) #micro meters
  dt=T/(Ntime) 	
  Ntime=2000
  N=2*Nhalf+1   #gesamte Anzahl von Grid-Punkten
  #definiere Faktor aus Gleichung (3)
  xi=1.0
  fc=dt/(xi*dx*dx)
  # definiere Gefäße
  rv=4   # Radius von Gefäßen in Mikrometer
  rd=5  # in Mikromater Radius plus Wanddick
  kv=0
  kd=0
  r30=1e20
  amaxgef =1.0  #Diffusionskoeffizient in der GefÃ¤ÃŸdicke
  #amax=260    #Diffusionskoeffizient
 
  amax=-1e20; 
  a =numpy.zeros((N+1,N+1)) 
  define_a(  a,    amax,   N,   Nhalf,   epsil1,   epsil2,   dx)
  dt_crit=dx*dx/(4.0*amax) # Stabilitätskriterium
  #Radien fuer Permeabilitaetsberechnung
  Rc=rv+1.0
  Rd=rd+1.0
  u0 = numpy.zeros((N+1,N+1))  
  u =numpy.zeros((N+1,N+1))  # numpy.random.normal(0.00000001,0.0000000000001,(N+1,N+1))
  c_ana = numpy.zeros((N+1,N+1))  #numpy.random.normal(0.00000001,0.00000001,(N+1,N+1))

  Sum_u    =numpy.zeros((Ntime+1)) 
  Sum_ana = numpy.zeros((Ntime+1)) 
  M_PI =math.pi;
  
  #define_a(a, amax, N, Nhalf, epsil1, epsil2, dx);
  counti =numpy.ones((N+1))
  countj=numpy.ones((N+1))
  countg=numpy.ones((N+1))
  countkdi=numpy.ones((N+1))#numpy.ones((N*N+2)) or #numpy.ones((N*N*N+2))
  countkdj=numpy.ones((N+1))#numpy.ones((N*N+2)) or #numpy.ones((N*N*N+2))
  countkdg=numpy.ones((N+1)) #numpy.ones((N*N+2)) or #numpy.ones((N*N*N+2))
  counti[0]=0
  countj[0]=0
 
  N_start = 0;
  Ntime_start = 0;
  N_end = 0;
  Ntime_end = 0;
  kv=0 ;
  kd=0 ;
  i30 = 0;
  j30 =0;
 
  Dif = 260.0;
  M = 16.60577881 / ( dx * dx);  #16.60577881 / (dx * dx);  

  #for i in range(0,N+1): 
  #  for j in range(0,N+1): 
        #u[i][j] = 0.000000000001+random.randint(1,1000); 
  #      u[i][j] = 0.000000000001+random.randint(1,1000)*0.000000000001; 
        #u[i][j] = 0.00000000000001   ; 
 
  u[Nhalf+1][Nhalf+1]=M;

  for i in range(1,N): 
    for j in range(1,N): 
      
        r=dx * math.sqrt( ((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))); 
        if(r<rv):
          kv=kv+1 
          counti[kv]=i 
          countj[kv]=j 
 
 
        if ((r>=rv) &  (r<rd)): 
          kd=kd+1;
          countkdi[kd]=i 
          countkdj[kd]=j 
 
 
        if(abs(r-30.0)<r30):   
          r30=r-30.0 
          i30=i 
          j30=j 


  dt_crit=dx*dx/(4.0*Dif) #8.0 * Dif 3d
  if(dt>dt_crit): 
    print(" None  " ,dt )
    exit()
  else:
    
    # Zeit-Schleife # here we start MPI steps
    Sum_u =  numpy.zeros((Ntime))
    Sum_u1 =numpy.zeros((Ntime))
    Sum_ana = numpy.zeros((Ntime))
    Sum_ana1 = numpy.zeros((Ntime))
    result = []
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #params = numpy.ones((N+1, N+1,N+1))    # parameters to send to calc_function
    #n = params.shape[0]

    #count = n // size  # number of catchments for each process to analyze
    #remainder = n % size  # extra catchments if n is not a multiple of size

 
    tag_data = 42
    tag_end = 23
    a = 0
    b = Ntime
    num_per_rank = b  // size
 
    lower_bound = a + rank * num_per_rank
    upper_bound = a + (rank + 1) * num_per_rank
   
    start_time = time.time()


    for it in range(lower_bound, upper_bound):

     #calc_function2D(it,N,u0,u,c_ana, dt,  fc, Nhalf,dx,Dif,M_PI, amax ,epsil2    )  # run the function for each parameter set and rank
     u0 = u.copy()
     #aif_0 = aif.copy() 
     # or  this
     #for l in range(1,N):  
     #  for m in range(1,N):  
     #     for k in range(1,N): 
     #       u0[l][m][k]=u[l][m][k];       
     #for l in range(1,kv):  
     #  aif_0[l]=aif[l]
       
     #hier wird beachtet, dass die Menge die weggeht von einem Ende des Quadrats, tritt wieder aus anderem Ende ein */
     # X-Schleife
     for i in range(1,N):  
       if(i==N):
        ip=1                    #ip=x+h
       else:
        ip=i+1
        
       if(i==1):
        im=N                    # im=x-h
       else:
        im=i-1
        
     # Y-Schleife
       for j in range(1,N): 
         if(j==N):
           jp=1        
         else:
          jp=j+1
       
         if(j==1):
           jm=N        
         else:
           jm=j-1                
           #ax_minus=0.5*(a[i][j][g]+a[im][j][g]);
           #ax_plus=0.5*(a[i][j][g]+a[ip][j][g]);
           #ay_minus=0.5*(a[i][j][g]+a[i][jm][g]);
	         #ay_plus=0.5*(a[i][j][g]+a[i][jp][g]);
  	       #az_minus=0.5*(a[i][j][g]+a[i][j][gm]);
  	       #az_plus=0.5*(a[i][j][g]+a[i][j][gp]); 

           # die Funktion
           # u[i][j][g]=u0[i][j][g]+fc*((0.5*(a[i][j][g]+a[im][j][g]))*u0[im][j][g]+(0.5*(a[i][j][g]+a[ip][j][g]))*u0[ip][j][g]+(0.5*(a[i][j][g]+a[i][jm][g]))*u0[i][jm][g]+(0.5*(a[i][j][g]+a[i][jp][g]))*u0[i][jp][g]+(0.5*(a[i][j][g]+a[i][j][gm]))*u0[i][j][gm]+(0.5*(a[i][j][g]+a[i][j][gp]))*u0[ip][j][gp]-((0.5*(a[i][j][g]+a[im][j][g]))+(0.5*(a[i][j][g]+a[ip][j][g]))+(0.5*(a[i][j][g]+a[i][jm][g]))+(0.5*(a[i][j][g]+a[i][jp][g])+(0.5*(a[i][j][g]+a[i][j][gm]))+(0.5*(a[i][j][g]+a[i][j][gp])))*u0[i][j][g]);              
         
         #u[i][j]=i+j # u0[i][j] +((fc*amax*epsil2)*(u0[im][j] +u0[ip][j] +u0[i][jm] +u0[i][jp] +u0[i][j] +u0[i][j] -(4.0)*u0[i][j] )); 
         tt = u0[i][j] + (((dt * Dif) / (dx * dx)) * (u0[im][j] + u0[ip][j] + u0[i][jm] + u0[i][jp] - 4.0 * u0[i][j]));
         #if (tt <0):
         #  tt= tt* -1
         u[i][j] =   tt 
       	 #für die Gefäßdicke , also definiere permeabilität
         tim = dt * it ;  
         if (tim == 0):
           tim = 1

           #c_ana[i][j][g] =2#(16.60577881 / (4.0 * M_PI * Dif * tim)) * math.exp(-(dx * dx * ( ((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))) / (4.0 * Dif * (tim))));
         c_ana[i][j] = (16.60577881 / (4.0 * M_PI * Dif * tim)) *  math.exp(-(dx * dx * (((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))) / (4.0 * Dif * (tim))));
 
     #calc_Sum_u2D(it,dx,N, u,c_ana,  Sum_u,Sum_ana )
     Sum_u[it] = 0;
     Sum_ana[it] = 0;
     for i in range(1,N): 
       for j in range(1,N): 

              Sum_u[it] = Sum_u[it] + u[i][j] ;
              Sum_ana[it] = Sum_ana[it] + c_ana[i][j];
     
     Sum_u[it] = Sum_u[it] * dx  * dx * (6.022e23); #Sum_u[it] * dx * dx * (6.022e23);
     Sum_ana[it] = Sum_ana[it] *  dx * dx * (6.022e23); #Sum_ana[it] * dx * dx * (6.022e23);
     
    #outputData = comm.gather(Sum_u,root=0)
     #print("This is processor ", rank, "and I am summing numbers from", lower_bound," to ", upper_bound - 1, flush=True)


 
    
    #comm.Reduce(local_results, local_results1, op=MPI.SUM, root=0)
    # collect the partial results and add to the total sum
    #comm.Reduce(summ, total, op=MPI.SUM, root=0)
    stop_time = time.time() 
    if rank == 0:
      c_ana1 = c_ana.copy()
      uu1 = u.copy()
      Sum_u1 =Sum_u.copy()
      Sum_ana1= Sum_ana.copy()
      for p in range(1, size): 
          received = numpy.empty(Ntime, dtype=float)
          received1 = numpy.empty(Ntime, dtype=float)  
          rank_size=  numpy.empty(1, dtype=int)
          #received2= numpy.empty(Ntime, dtype=float)
          comm.Recv(received, p, tag=tag_data)
          comm.Recv(received1, p, tag=tag_data)
          comm.Recv(c_ana1, p, tag=tag_data)
          comm.Recv(uu1, p, tag=tag_data)
          comm.Recv(rank_size, p, tag=tag_data)
          lower_bound = a+  rank_size[0] * num_per_rank
          upper_bound = a+ (rank_size[0] + 1) * num_per_rank
          #c_ana1 = received2 
          for k in  range(lower_bound, upper_bound):
             
               Sum_u1[k]= received[k]
               Sum_ana1[k]= received1[k] 
              
    else:
      lower_bound1 = numpy.empty(1, dtype=int)
      lower_bound1[0] = rank
      comm.Send(Sum_u, dest=0, tag=tag_data)
      comm.Send(Sum_ana, dest=0, tag=tag_data)
      comm.Send(c_ana, dest=0, tag=tag_data)      
      comm.Send(u, dest=0, tag=tag_data)
      comm.Send(lower_bound1, dest=0, tag=tag_data)
      comm.send(None, dest=0, tag=tag_end)
 
    #comm.Barrier()

    #rank_size = -1
    #if rank > 0:
    #  comm.Send(  Sum_u, dest=0, tag=14)  # send results to process 0
    # N_start = 0;
    # Ntime_start = 0;
    # N_end = 0;
    #Ntime_end = 0;
    Ntime_start =lower_bound;
    N_start =lower_bound;
    if rank == 0:
      Sum_uf = open("d:/Sum_uf.txt", "w")
      Sum_anaf = open("d:/Sum_anaf.txt", "w")
      uf = open("d:/uf.txt", "w")
      c_anaf = open("d:/c_ana.txt", "w")
      
      print(" Sum_u = " , Sum_u1)
      print(" Sum_ana = " , Sum_ana1)
      print(" u = " , u)      
      print(" c_ana = " , c_ana1)
      for it in range(1,Ntime): 
        Sum_uf.write( str((it + Ntime_start) * dt) +  "    " +str( Sum_u1[it]) +"\n")
        Sum_anaf.write(str((it + Ntime_start) * dt)+ "    " + str (Sum_ana1[it]) +"\n")

      for i in range(1,N): 
          for j in range(1,N): 
                uf.write( str((i + N_start) * dx) +  "    " + str((j + N_start) * dx) +  "    " + str( u[i][j]) +"\n")
                c_anaf.write( str((i + Ntime_start) * dx) +  "    " +  "    " + str((j + N_start) * dx) +  "    " + str( c_ana[i][j]) +"\n")
      Sum_uf.close()
      Sum_anaf.close()
      uf.close()
      c_anaf.close()
    if rank == 0:

      #draw fig 1,good
      #x = numpy.linspace(0, 20000  , N+1)
      #y = numpy.linspace(0, 2000 ,N+1)
      #ax.set_xlim(100, 2000)
      #ax.set_ylim(100, 2000)
      #ax.set_zlim(0.00000001,  0.0000000000000000000001)

      # draw fig 2,not good
      # ax.set_zlim(0.00000001,  0.0000000000000000000001)
      #x = numpy.linspace(1000, 25000  , N+1)
      #y = numpy.linspace(1000, 2500 ,N+1)

      # this good 
      #x = numpy.linspace(0, 10, N+1)
      #y = numpy.linspace(0, 10, N+1)

      x = numpy.linspace(1, 5, N+1)
      y = numpy.linspace(1, 5, N+1)

      X, Y = numpy.meshgrid(x, y)

  
      fig = pyplot.figure()
      ax = fig.gca(projection='3d')
      #ax.set_xlim(0, 1500)
      #ax.set_ylim(0, 1500)
      ax.set_zlim(numpy.amin(u)    , numpy.amax(u)*2 )
   
      surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
          linewidth=0, antialiased=True)
      ax.view_init(elev=30, azim=220)

      ax.set_xlabel('$x$')
      ax.set_ylabel('$y$');

 
      pyplot.show()
      #print(" uu1 = " , uu1)
  

def define_a(  a,    amax,   N,   Nhalf,   epsil1,   epsil2,   dx): 
  R =0;
  epsil  =0
  Rc=99.7356; # Radius fÃ¼r die HÃ¤lfte des Volumens
  D=260.0;
  bet=0.24926;  
  for i in range(1,N): 
   for j in range(1,N): 
     for g in range(1,N):
      R=dx*math.sqrt( ((i-Nhalf-1)*(i-Nhalf-1)+(j-Nhalf-1)*(j-Nhalf-1)));
      if(R<Rc):
        epsil=epsil1;         # inneres Gebiet, hier Ã¤ndert sich epsil mit Schritten von 0.05    
      else:
        epsil=epsil2;         #    Ã¤uÃŸeres Gebiet, hier Ã¤ndert sich epsil mit Schritten von 0.1
    
      # definiere a
      if(epsil <=0.3):
       a[i][j][g]=D*(pow(epsil,bet+1.0));   
      else:
       a[i][j][g]=D*(2.0*epsil/(3.0-epsil));
     
      if(a[i][j][g]>amax):
       amax=a[i][j][g];
          
def define_a2d(  a,    amax,   N,   Nhalf,   epsil1,   epsil2,   dx): 
  R =0;
  epsil  =0
  Rc=99.7356; # Radius fÃ¼r die HÃ¤lfte des Volumens
  D=260.0;
  bet=0.24926;  
  for i in range(1,N): 
   for j in range(1,N): 

    R=dx*math.sqrt( ((i-Nhalf-1)*(i-Nhalf-1)+(j-Nhalf-1)*(j-Nhalf-1)));
    if(R<Rc):
      epsil=epsil1;         # inneres Gebiet, hier Ã¤ndert sich epsil mit Schritten von 0.05    
    else:
      epsil=epsil2;         #    Ã¤uÃŸeres Gebiet, hier Ã¤ndert sich epsil mit Schritten von 0.1
    
    # definiere a
    if(epsil <=0.3):
     a[i][j]=D*(pow(epsil,bet+1.0));   
    else:
     a[i][j]=D*(2.0*epsil/(3.0-epsil));
     
    if(a[i][j]>amax):
     amax=a[i][j];
def drwadiffuionMPI3DnoMPI():
 
  epsil1 = 1.0;
  epsil2 = 0.8;
  L = 300.0;
  Nhalf = 50;
  T=50.0;
  Ntime =20;
  dx = L /  (2 * Nhalf);
  dt = T /  (100000); #T / double(Ntime); 
  N = 2 * Nhalf + 1;
  Dif = 260.0;
  M=16.60577881/(dx*dx*dx); 
  #M = 16.60577881/(dx*dx)# 16.60577881 / (dx * dx * dx);
 
  M_PI =math.pi;

  amax=260.0 # amax=-1e20; 
  amaxgef =1.0 

  xi=1.0;
  fc=dt/(xi*dx*dx) 
 
  #definiere Gefäße
  rv=40.0 # in Mikrometer
  rd=50.0  #Radius plus Dicke
  kv=0
  kd=0
  r30=1e20
    
  a =numpy.zeros((N+1,N+1,N+1))
  define_a(a, amax, N, Nhalf, epsil1, epsil2, dx)
  Rc=rv+1.0
  Rd=rd+1.0
  u0 =numpy.zeros((N+1, N+1,N+1))
  u =numpy.zeros((N+1, N+1,N+1))
  c_ana =numpy.zeros((N+1, N+1,N+1))
  counti =numpy.zeros((N*N*N))
  countj=numpy.zeros((N*N*N))
  countg=numpy.zeros((N*N*N))
 
  counti[0]=0
  countj[0]=0
  countg[0]=0

  N_start = 0;
  Ntime_start = 0;
  N_end = 0;
  Ntime_end = 0;
  kv=0 ;
  kd=0 ;
  i30 = 0;
  j30 =0;
 
  Dif = 260.0; 
 
  u[Nhalf+1][Nhalf+1][Nhalf+1]=M;

  for i in range(1,N): 
    for j in range(1,N): 
      for g in range(1,N): 
        r=dx * math.sqrt( ((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1) + (g - Nhalf - 1) * (g - Nhalf - 1))) 
        if(r<rv):
          kv=kv+1 
          counti[kv]=i 
          countj[kv]=j 
          countg[kv] = g
          
        if(abs(r-30.0)<r30):   
          r30=r-30.0 
          i30=i 
          j30=j 
          g30 = g

  dt_crit = dx * dx / (8.0 * Dif); #8.0 * Dif 3d #dx * dx / (8.0 * Dif);
 
  if(dt>dt_crit): 
    print(" None  " ,dt )
    exit()
  else:    
    # Zeit-Schleife # here we start MPI steps
    Sum_u =  numpy.zeros((Ntime))
    Sum_u1 =numpy.zeros((Ntime))
    Sum_ana = numpy.zeros((Ntime))
    Sum_ana1 = numpy.zeros((Ntime))
 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

  
    tag_data = 42
    tag_end = 23
    a = 0
    b = Ntime
    num_per_rank = b  // size
 
    lower_bound = a + rank * num_per_rank
    upper_bound = a + (rank + 1) * num_per_rank
   
    start_time = time.time()


    for it in range(lower_bound, upper_bound):

     #calc_function2D(it,N,u0,u,c_ana, dt,  fc, Nhalf,dx,Dif,M_PI, amax ,epsil2    )  # run the function for each parameter set and rank
     u0 = u.copy()
     #aif_0 = aif.copy() 
     # or  this
     #for l in range(1,N):  
     #  for m in range(1,N):  
     #     for k in range(1,N): 
     #       u0[l][m][k]=u[l][m][k];       
     #for l in range(1,kv):  
     #  aif_0[l]=aif[l]
       
     #hier wird beachtet, dass die Menge die weggeht von einem Ende des Quadrats, tritt wieder aus anderem Ende ein */
     # X-Schleife
     for i in range(1,N):  
       if(i==N):
        ip=1                    #ip=x+h
       else:
        ip=i+1
        
       if(i==1):
        im=N                    # im=x-h
       else:
        im=i-1
        
     # Y-Schleife
       for j in range(1,N): 
         if(j==N):
           jp=1        
         else:
          jp=j+1
       
         if(j==1):
           jm=N        
         else:
           jm=j-1                
           #ax_minus=0.5*(a[i][j][g]+a[im][j][g]);
           #ax_plus=0.5*(a[i][j][g]+a[ip][j][g]);
           #ay_minus=0.5*(a[i][j][g]+a[i][jm][g]);
	         #ay_plus=0.5*(a[i][j][g]+a[i][jp][g]);
  	       #az_minus=0.5*(a[i][j][g]+a[i][j][gm]);
  	       #az_plus=0.5*(a[i][j][g]+a[i][j][gp]); 

           # die Funktion
           # u[i][j][g]=u0[i][j][g]+fc*((0.5*(a[i][j][g]+a[im][j][g]))*u0[im][j][g]+(0.5*(a[i][j][g]+a[ip][j][g]))*u0[ip][j][g]+(0.5*(a[i][j][g]+a[i][jm][g]))*u0[i][jm][g]+(0.5*(a[i][j][g]+a[i][jp][g]))*u0[i][jp][g]+(0.5*(a[i][j][g]+a[i][j][gm]))*u0[i][j][gm]+(0.5*(a[i][j][g]+a[i][j][gp]))*u0[ip][j][gp]-((0.5*(a[i][j][g]+a[im][j][g]))+(0.5*(a[i][j][g]+a[ip][j][g]))+(0.5*(a[i][j][g]+a[i][jm][g]))+(0.5*(a[i][j][g]+a[i][jp][g])+(0.5*(a[i][j][g]+a[i][j][gm]))+(0.5*(a[i][j][g]+a[i][j][gp])))*u0[i][j][g]);                                
 
         for g in range(1,N):
           if (g == N):
              gp = 1                    
           else: 
              gp = g + 1;                    
           if (g == 1):
              gm = N                     
           else:
              gm = g - 1
                    
           u[i][j][g] = u0[i][j][g] + (((dt * Dif) / (dx * dx)) * (u0[im][j][g] + u0[ip][j][g] + u0[i][jm][g] + u0[i][jp][g] + u0[i][j][gm] + u0[i][j][gp] - 8.0 * u0[i][j][g]));
           
       	   #für die Gefäßdicke , also definiere permeabilität
           tim = dt * it ;  
           if (tim == 0):
             tim = 1

             #c_ana[i][j][g] =2#(16.60577881 / (4.0 * M_PI * Dif * tim)) * math.exp(-(dx * dx * ( ((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1))) / (4.0 * Dif * (tim))));
           #c_ana[i][j][g] =(16.60577881 * 3 / (4.0 * M_PI * dx * dx * dx)) * (1 - math.exp(-(dx * dx * (((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1)
           #                                                                                         + (g - Nhalf - 1) * (g - Nhalf - 1)))) / (4.0 * Dif * (tim))))
           temps =(16.60577881 * 3 / (4.0 * M_PI * dx * dx * dx)) * (1 - math.exp(-(dx * dx * (((i - Nhalf - 1) * (i - Nhalf - 1) + (j - Nhalf - 1) * (j - Nhalf - 1)
                                                                                                     + (g - Nhalf - 1) * (g - Nhalf - 1)))) / (4.0 * Dif * (tim))))
           c_ana[i][j][g] =temps

     #calc_Sum_u2D(it,dx,N, u,c_ana,  Sum_u,Sum_ana )
     Sum_u[it] = 0;
     Sum_ana[it] = 0;
     for i in range(1,N): 
       for j in range(1,N): 
         for g in range(1,N): 
              Sum_u[it] = Sum_u[it] + u[i][j][g] ;
              Sum_ana[it] = Sum_ana[it] + c_ana[i][j][g];
     
     Sum_u[it] = Sum_u[it] * dx * dx * dx * (6.022e23)   #Sum_u[it] * dx * dx * (6.022e23);
     Sum_ana[it] = Sum_ana[it] * dx * dx * dx * (6.022e23)   #Sum_ana[it] * dx * dx * (6.022e23);
     
  
    
    #comm.Reduce(local_results, local_results1, op=MPI.SUM, root=0)
    # collect the partial results and add to the total sum
    #comm.Reduce(summ, total, op=MPI.SUM, root=0)
    stop_time = time.time() 
    if rank == 0:
      c_ana1 = c_ana.copy()
      uu1 = u.copy()
      Sum_u1 =Sum_u.copy()
      Sum_ana1= Sum_ana.copy()
      for p in range(1, size): 
          received = numpy.empty(Ntime, dtype=float)
          received1 = numpy.empty(Ntime, dtype=float)  
          rank_size=  numpy.empty(1, dtype=int)
          #received2= numpy.empty(Ntime, dtype=float)
          comm.Recv(received, p, tag=tag_data)
          comm.Recv(received1, p, tag=tag_data)
          comm.Recv(c_ana1, p, tag=tag_data)
          comm.Recv(uu1, p, tag=tag_data)
          comm.Recv(rank_size, p, tag=tag_data)
          lower_bound = a+  rank_size[0] * num_per_rank
          upper_bound = a+ (rank_size[0] + 1) * num_per_rank
          #c_ana1 = received2 
          for k in  range(lower_bound, upper_bound):
             
               Sum_u1[k]= received[k]
               Sum_ana1[k]= received1[k] 
              
    else:
      lower_bound1 = numpy.empty(1, dtype=int)
      lower_bound1[0] = rank
      comm.Send(Sum_u, dest=0, tag=tag_data)
      comm.Send(Sum_ana, dest=0, tag=tag_data)
      comm.Send(c_ana, dest=0, tag=tag_data)      
      comm.Send(u, dest=0, tag=tag_data)
      comm.Send(lower_bound1, dest=0, tag=tag_data)
      comm.send(None, dest=0, tag=tag_end)  

    #rank_size = -1
    #if rank > 0:
    #  comm.Send(  Sum_u, dest=0, tag=14)  # send results to process 0
    # N_start = 0;
    # Ntime_start = 0;
    # N_end = 0;
    #Ntime_end = 0;
 
    if rank == 0:
      Sum_uf = open("d:/Sum_uf.txt", "w")
      Sum_anaf = open("d:/Sum_anaf.txt", "w")
      uf = open("d:/uf.txt", "w")
      c_anaf = open("d:/c_ana.txt", "w")
      
      #print(" Sum_u = " , Sum_u1)
      #print(" Sum_ana = " , Sum_ana1)
      #print(" u = " , u)      
      #print(" c_ana = " , c_ana1)
      print(" done"  ) 
      for it in range(1,Ntime): 
        Sum_uf.write( str((it + N) * dt) +  "    " +str( Sum_u1[it]) +"\n")
        Sum_anaf.write(str((it + N) * dt)+ "    " + str (Sum_ana1[it]) +"\n")

      for i in range(1,N): 
          for j in range(1,N): 
            for g in range(1,N): 
                uf.write( str((i + N) * dx) +  "    " + str((j + N) * dx) +  "    " + str((g + N) * dx) +  "    " + str( u[i][j][g]) +"\n")
                c_anaf.write( str((i + N) * dx) +  "    "    + str((j + N) * dx) +  "    " + str((g + N) * dx) +  "    "  + str( c_ana[i][j][g]) +"\n")
      Sum_uf.close()
      Sum_anaf.close()
      uf.close()
      c_anaf.close()
    if rank ==0:

      #draw fig 1,good
      #x = numpy.linspace(0, 20000  , N+1)
      #y = numpy.linspace(0, 2000 ,N+1)
      #ax.set_xlim(100, 2000)
      #ax.set_ylim(100, 2000)
      #ax.set_zlim(0.00000001,  0.0000000000000000000001)

      # draw fig 2,not good
      # ax.set_zlim(0.00000001,  0.0000000000000000000001)
      #x = numpy.linspace(1000, 25000  , N+1)
      #y = numpy.linspace(1000, 2500 ,N+1)

      # this good 
      Data = numpy.genfromtxt( "d:\\uf.txt" )
      x=Data[:,0] 
      y =Data[:,1] 
      val=Data[:,3]

      fig = plt.figure(figsize =(14, 9))
      ax = plt.axes(projection ='3d')  
      # Creating plot
      ax.scatter3D(x, y, val)  
      # show plot
      plt.show()

      #x = numpy.linspace(0, 10, N+1)
      #y = numpy.linspace(0, 10, N+1)
      #z = numpy.linspace(0, 10, N+1)
 
 
      #Plotting
      #fig = pyplot.figure()
      #X4, Y4, Z4 = numpy.meshgrid(x, y,z)
      #ax = fig.add_subplot(222, projection='3d')
      #img = ax.scatter(X4, Y4, Z4, c=u, cmap=pyplot.jet())
      #fig.colorbar(img)

 
      #pyplot.show()

      #print(" uu1 = " , uu1)
def plotfile():
      data=numpy.loadtxt('d:\\uf.txt')  
      pyplot.plot(data[:,0],data[:,1],data[:,2],'bo')  

      X=data[:,0]  
      Y=data[:,1]  
      Z=data[:,2] 
     

      x = numpy.linspace(1, 50, 100)
      y = numpy.linspace(1, 50, 100)
      X, Y = numpy.meshgrid(x, y)

      fig = pyplot.figure()
      ax = fig.gca(projection='3d')
      #ax.set_xlim(0, 1500)
      #ax.set_ylim(0, 1500)
      #ax.set_zlim(0.00000000001 , 0.000000001 )
   
      surf = ax.plot_surface(X, Y, data, rstride=1, cstride=1, #cmap=cm.viridis,
          linewidth=0, antialiased=True)
 
      ax.set_xlabel('$x$')
      ax.set_ylabel('$y$');

def plotfile2():
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    data = numpy.genfromtxt('d:\\uf.txt')  
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    #xi = numpy.linspace(min(x), max(x),0.1)
    #yi = numpy.linspace(min(y), max(y),0.1)

    xi = numpy.linspace(1, 50, 100)
    yi = numpy.linspace(1, 50, 100)

    X, Y = numpy.meshgrid(xi, yi)
    Z = griddata(x, y, z, xi, yi )

    ax.set_xlabel('$x$', fontsize = 14)
    ax.set_ylabel('$y$', fontsize = 14)
    ax.set_zlabel('$z$', fontsize = 14)
    ax.set_title('result..', fontsize = 14)

    surf = ax.plot_surface(X, Y, Z, linewidth=1, antialiased=True)

    ax.set_zlim3d(0,1)

    pyplot.show()  
 

def plotfile3():
 
    npts = 200
    x = numpy.random.uniform(-2, 2, npts)
    y = numpy.random.uniform(-2, 2, npts)
    z = x*numpy.exp(-x**2 - y**2)
    # define grid.
    xi = numpy.linspace(-2.1, 2.1, 100)
    yi = numpy.linspace(-2.1, 2.1, 200)
    # grid the data.
    zi = griddata(x, y, z, xi, yi )
    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(xi, yi, zi, 15,
                      vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(x, y, marker='o', s=5, zorder=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('griddata test (%d points)' % npts)
    plt.show()


def plotfile4():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
 
    data = numpy.genfromtxt('d:\\uf.txt') 
    #zero coupon maturity dates
    y = data[:,0]
    #tenor
    x = data[:,1]
    #rates
    z = data[:,2]
 
    #maturity dates chart axis
    uniquemat = numpy.unique(y)
 
    #the zc rate maturity axis is arranged in log space
    yi = numpy.unique(y)
 
    #tenor chart axis
    xi = numpy.unique(x)
 
    X, Y = numpy.meshgrid(xi, yi)
    #Z = il.griddata(x, y, z, xi, yi,(X, Y),   interp='linear') 

    Z = il.griddata( (x, y,z), data, (xi, yi),  method= 'linear' )  
 
    # Plot rate surface
    
    fig = plt.figure()
    fig.suptitle('EUR rate surface',fontsize=20)
    ax = fig.gca(projection='3d')
 
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,alpha=0.3)
    ax.set_xlabel('Tenor')
    ax.set_ylabel('Zc maturity')
    ax.set_zlabel('Zc Rate (%)')
 
    # Override tenor axis labels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '1M'
    labels[1] = '3M'
    labels[2] = '6M'
    labels[4] = '1Y'
 
    ax.set_xticklabels(labels)
 
    # Plot 3D contour
    zzlevels = numpy.linspace(Z.min(),Z.max(),num=8,endpoint=True)
    xxlevels = numpy.linspace(X.min(),X.max(),num=8,endpoint=True)
    yylevels = numpy.linspace(Y.min(),Y.max(),num=8,endpoint=True)
    cset = ax.contour(X, Y, Z, zzlevels, zdir='z',offset=Z.min(), cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, xxlevels, zdir='x',offset=X.min(), cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, yylevels, zdir='y',offset=Y.max(), cmap=cm.coolwarm)
 
    plt.clabel(cset,fontsize=10, inline=1)
 
    ax.set_zlim3d(numpy.min(Z), numpy.max(Z))
 
    plt.show()

#this goood
def plotfile5aa():
      Data = numpy.genfromtxt( 'd:\\‏‏‏‏u_ana2.dat'  )

      x = numpy.linspace(1,10,num=8,endpoint=True)
      y = numpy.linspace(Data[:,0].min(),Data[:,1].max(),num=8,endpoint=True)
 
      X, Y   = numpy.meshgrid(x,y   )

    #  Z = Data[:,2]
      # get 2D z data
      #Z = numpy.loadtxt("datacsv_2d_Z.csv", delimiter=",")
      Z =Data[:,2]# numpy.tile(Data[:,2], (len(Data[:,2]), 1))

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

      ax.plot_surface(X, Y, Z, cmap='ocean')

      plt.show()

def plotfile5():
      Data = numpy.genfromtxt( 'd:\\‏‏c_ana.dat'  )

      # create 2d x,y grid (both X and Y will be 2d)
      X, Y   = numpy.meshgrid(Data[:,0], Data[:,1]   )

    #  Z = Data[:,2]
      # get 2D z data
      #Z = numpy.loadtxt("datacsv_2d_Z.csv", delimiter=",")
      Z = numpy.tile(Data[:,2], (len(Data[:,2]), 1))

      fig = pyplot.figure()
      ax = fig.add_subplot(111, projection='3d')

      ax.plot_surface(X, Y, Z, cmap='ocean')

      pyplot.show()

def plotfile5a():
      Data = numpy.genfromtxt( 'd:\\‏‏‏‏u_ana2.dat'  )

      x=Data[:,0] 
      y =Data[:,1] 
      z =Data[:,2] 
      val=Data[:,3] 

     

      fig = plt.figure(figsize =(14, 9))
      ax = plt.axes(projection ='3d')
  
      # Creating plot
      ax.scatter3D(x, y, val)
  
      # show plot
      plt.show()


def plotfile5temp():
  x = numpy.outer(numpy.linspace(-3, 3, 32), numpy.ones(32))
  y = x.copy().T # transpose
  z = (numpy.sin(x **2) + numpy.cos(y **2) )
  
  # Creating figyre
  fig = plt.figure(figsize =(14, 9))
  ax = plt.axes(projection ='3d')
  
  # Creating plot
  ax.plot_surface(x, y, z)
  
  # show plot
  plt.show()
def plotfile6():
      Data = numpy.genfromtxt( 'd:\\‏‏c_ana.dat'  )

      # create 2d x,y grid (both X and Y will be 2d)
      x=Data[:,0] 
      y =Data[:,1] 
      z =Data[:,2] 
      val=Data[:,3] 
 

 
 
      #Plotting
      fig = pyplot.figure()
 
      ax = fig.add_subplot(111, projection='3d')
      img = ax.scatter(z, y, val, c=z, cmap=pyplot.jet())
      fig.colorbar(img)

 
      pyplot.show()
    #  print("Sum_u= " ,Sum_u )
 
    #u=final_results.copy()

   # for  p in range(1,kd): 
   #  u[countkdi[p]][countkdj[p]][countkdg[p]] = u0[countkdi[p]][countkdj[p]][countkdg[p]]+((fc*amaxgef*epsil2)*(u0[countkdi[p]-1][countkdj[p]][countkdg[p]]+u0[countkdi[p]+1][countkdj[p]][countkdg[p]]+u0[countkdi[p]][countkdj[p]-1][countkdg[p]]+u0[countkdi[p]][countkdj[p]+1][countkdg[p]]+u0[countkdi[p]][countkdj[p]][countkdg[p]-1]+u0[countkdi[p]][countkdj[p]][countkdg[p]+1]-(4.0)*u0[countkdi[p]][countkdj[p]][countkdg[p]]));
 


# graphics output section
 
#computeNexttemp(4)

#firstdomain()
#diff4d()
#diffuse2(4)
#diffuse3(4)
#diffuse4(4)
#diffuse43d(4)
#diffuse53d(4)
#diffuse4(4)
#computeNexttemp(4)
#calcC(30)
#drwadiffuion()
#MPI4py()
#mpigood()
#my_MPi1()
#drwadiffuionMPI()
#drwadiffuionMPI2D()
#drwadiffuionMPI2DnoMPI() # this good final
drwadiffuionMPI3DnoMPI() # this good
#plotfile()
#plotfile2()
#plotfile3()
#plotfile4()
#plotfile5()
##########plotfile5a()
#plotfile5aa()
#plotfile6()
#newarray()
#mains()
#mpigood2()
#drawmatrix()
#pyplot.show()
 
