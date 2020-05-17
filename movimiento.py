import numpy as np
from obspy.core import Trace,Stream,UTCDateTime,Stats
import matplotlib.pyplot as plt

from pyrotd import calc_spec_accels
import pyrotd
from numpy.fft import rfft, rfftfreq
import obspy.signal.util

import obspy
from obspy.signal.detrend import polynomial,spline
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing #FFT suavizada con el método de konno_ohmachi

import scipy
from scipy import integrate
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift

import os

class MovimientoFuerte():
    """
    Determina parametros de movimiento fuerte de una senal de aceleraciones 

    Atributos
    ---------
    file : str
        nombre del archivo .anc con el registro de aceleraciones
    sismo : str
        informacion del sismo de origen
    lat_evento : float
        latitud del epicentro sismo
    long_evento : float
        longitud del epicentro del sismo
    profundidad_evento : float
        profundidad del epicentro del sismo
    codigo_estacion : str
        codigo de la estacion sismologica
    estacion: str
        nombre de la ubicacion de la estacion
    lat_estacion : float
        latitud de la estacion sismologica
    long_estacion : float
        longitud de la estacion sismologica
    muestreo : float
        numero de muestras por segundo 
    numero_datos: float
        numero de muestras total del registro
    duracion : float
        duracion en segundos del registro

    Metodos
    ------------


    """
    def __init__(self,file,correccion):

        path = 'Quetame/20080524192044_anc/subdirectorio/'

        '''Inicializa con valores del file .anc'''
        self.file = path + file
        self.correccion = correccion
        archivo = open(self.file,"r")
        linea = archivo.readlines()
        self.sismo = linea[1]
        self.lat_evento = float(linea[2][28:35])
        self.long_evento = float(linea[3][28:35]) 
        self.profundidad_evento = float(linea[4][28:33])
        self.codigo_estacion = linea[5][23:28]
        self.estacion = linea[6]
        self.lat_estacion = float(linea[7][32:45])
        self.long_estacion = float(linea[8][34:45])
        self.dist_epicentral = float(linea[9][22:28])
        self.dist_hipocentral = float(linea[10][23:29])
        self.muestreo = float(linea[11][34:41])
        self.numero_datos = int(linea[12][17:21])
        self.duracion = float(linea[13][21:41])
        
        component_1 = linea[19][0:15].split()[0]
        component_2 = linea[19][20:36].split()[0]
        component_3 = linea[19][40:56].split()[0]
        components = {}
        
        if component_1 == 'NS':
            components['NS']=0
        else:
            if component_1 =='EW':
                components['EW']=0      
            if component_1 == 'VER':
                components['VER']=0
        if component_2 == 'NS':
            components['NS']=1
        else:
            if component_2 =='EW':
                components['EW']=1
            if component_2 == 'VER':
                components['VER']=1
        if component_3 == 'NS':
            components['NS']=2
        else:
            if component_3 =='EW':
                components['EW']=2
            if component_3 == 'VER':
                components['VER']=2

        self.EW_o = np.array([])
        self.VER_o = np.array([])
        self.NS_o = np.array([])
        
        for line in linea[20:len(linea)]:
            
            self.NS_line = float(line.split()[components['NS']])/100
            self.VER_line = float(line.split()[components['VER']])/100
            self.EW_line = float(line.split()[components['EW']])/100
            
            self.EW_o = np.append(self.EW_o, self.EW_line)
            self.VER_o = np.append(self.VER_o, self.VER_line)
            self.NS_o = np.append(self.NS_o, self.NS_line)
        
        if correccion == 'crudo':
            self.EW = self.EW_o
            self.VER = self.VER_o
            self.NS = self.NS_o
        else:

            if correccion == 'lineal':
                self.VER = scipy.signal.detrend(self.VER_o, axis=-1, type='linear', bp=0)
                self.NS = scipy.signal.detrend(self.NS_o, axis=-1, type='linear', bp=0)
                self.EW = scipy.signal.detrend(self.EW_o, axis=-1, type='linear', bp=0)
            else:
                if correccion == 'orden2':
                    self.VER = polynomial(self.VER_o, order=2)
                    self.NS = polynomial(self.NS_o, order=2)
                    self.EW = polynomial(self.EW_o, order=2)
                else:
                    if correccion == 'spline':
            
                        self.VER = spline(self.VER_o, order=2, dspline=1000)  
                        self.NS = spline(self.NS_o, order=2, dspline=1000)
                        self.EW = spline(self.EW_o, order=2, dspline=1000)
                    else:
                        print("Incluya correccion tipo: 'crudo','lineal','orden2','spline'")
        
        self.tiempo = np.arange(0,self.duracion,self.muestreo)
        self.tiempoV = self.tiempo[0:len(self.tiempo)-1]
        self.tiempoD = self.tiempoV[0:len(self.tiempoV)-1]

        self.vel_VER = integrate.cumtrapz(self.VER, self.tiempo, self.muestreo, 0)
        self.vel_NS = integrate.cumtrapz(self.NS, self.tiempo, self.muestreo, 0)
        self.vel_EW = integrate.cumtrapz(self.EW, self.tiempo, self.muestreo, 0)
                                    
        self.desp_VER = integrate.cumtrapz(self.vel_VER, self.tiempoV, self.muestreo, 0)
        self.desp_NS = integrate.cumtrapz(self.vel_NS, self.tiempoV, self.muestreo, 0)
        self.desp_EW = integrate.cumtrapz(self.vel_EW, self.tiempoV, self.muestreo, 0)
        
        '''Informacion colores plot'''
        self.infplot = {'NS':{'Color':'green','label':'NS'},
                        'EW':{'Color':'orange','label':'EW'},
                         'VER':{'Color':'blue','label':'VER'}}



    def plot(self,x_plot,y_plot,x_label,y_label):
        """
        Crea un formato de plot para los parametros determinados

        """

        #Valores maximos y minimos del plot
        max_ploty = max(1.1 * np.max(y_plot), abs(1.1*np.min(y_plot)))

        if np.min(y_plot)>0.000000000001:
            min_plot = 0
        else:
            min_plot= -1 * max_ploty 
      
        plt.figure(figsize=(10,4))

        plt.subplot(1,3,1)
        plt.plot(x_plot[0],y_plot[0],label=self.infplot['NS']['label'],color=self.infplot['NS']['Color'])
        plt.ylim(min_plot,max_ploty)
        plt.grid()
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(x_plot[1],y_plot[1],label=self.infplot['EW']['label'],color=self.infplot['EW']['Color'])
        plt.ylim(min_plot,max_ploty)
        plt.xlabel(x_label)
        plt.grid()
        plt.legend()
        
        plt.subplot(1,3,3)
        plt.plot(x_plot[2],y_plot[2],label=self.infplot['VER']['label'],color=self.infplot['VER']['Color'])
        plt.ylim(min_plot,max_ploty)
        plt.xlabel(x_label)
        plt.grid()
        plt.legend()

        plt.show()

    def plotGeneral(self,x_plot,y_plot,x_label,y_label,xplot_max):
        """
        Crea un formato de plot general para los parametros determinados

        """
        #Valores maximos y minimos del plot
        max_ploty = max(1.1 * np.max(y_plot), abs(1.1*np.min(y_plot)))

        if np.min(y_plot)>0.000000000001:
            min_plot = 0
        else:
            min_plot= -1 * max_ploty 
      
        plt.figure(figsize=(10,4))

        plt.subplot(1,3,1)
        plt.plot(x_plot[0],y_plot[0],label=self.infplot['NS']['label'],color=self.infplot['NS']['Color'])
        plt.ylim(min_plot,max_ploty)
        plt.xlim(0,xplot_max)
        plt.grid()
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(x_plot[1],y_plot[1],label=self.infplot['EW']['label'],color=self.infplot['EW']['Color'])
        plt.ylim(min_plot,max_ploty)
        plt.xlim(0,xplot_max)
        plt.xlabel(x_label)
        plt.grid()
        plt.legend()
        
        plt.subplot(1,3,3)
        plt.plot(x_plot[2],y_plot[2],label=self.infplot['VER']['label'],color=self.infplot['VER']['Color'])
        plt.ylim(min_plot,max_ploty)
        plt.xlim(0,xplot_max)
        plt.xlabel(x_label)
        plt.grid()
        plt.legend()

        plt.show()
      
    def plot_acel(self):

        self.tiempoX = [self.tiempo,self.tiempo,self.tiempo]
        self.aceleracionY = [self.NS,self.EW,self.VER]
                
        self.plot(self.tiempoX,self.aceleracionY,'Tiempo, [s]', 'Aceleracion [$m/s^2$]')

    def plot_vel(self):

        self.tiempoVelocidad = [self.tiempoV,self.tiempoV,self.tiempoV]
        self.velocidadY = [self.vel_NS,self.vel_EW,self.vel_VER]

        self.plot(self.tiempoVelocidad,self.velocidadY,'Tiempo, [s]', 'Velocidad [$m/s$]')

    def plot_desp(self):

        self.tiempo_desp = [self.tiempoD,self.tiempoD,self.tiempoD]
        self.desplazamientoY = [self.desp_NS,self.desp_EW,self.desp_VER]

        self.plot(self.tiempo_desp,self.desplazamientoY,'Tiempo, [s]', 'Desplazamiento, [m]')
       
    def val_pico(self):    
       
        # Parametros maximos
        self.PGA = np.max(np.abs(self.NS))
        self.PGV = np.max(np.abs(self.vel_NS))
        self.PGD = np.max(np.abs(self.desp_NS))
        
        self.ratioAV = self.PGA / self.PGV
        self.newmark_p = (self.PGA*self.PGD)/ self.PGV**2
        
    def IA(self):
        
        # Funciones de integracion a(t)**2

        y_integrate_NS = self.NS**2
        y_integrate_EW = self.EW**2
        y_integrate_VER = self.VER**2
        
        factor_IA = np.pi/(2*9.8)
        
        # Se integra con la ley del trapecio
        arias_inte_NS = integrate.cumtrapz(y_integrate_NS,self.tiempo,)
        arias_inte_EW = integrate.cumtrapz(y_integrate_EW,self.tiempo)
        arias_inte_VER = integrate.cumtrapz(y_integrate_VER,self.tiempo)
        
        #Se determina la intensidad de Arias
        arias_NS = factor_IA * arias_inte_NS
        arias_EW = factor_IA * arias_inte_EW
        arias_VER = factor_IA * arias_inte_VER
        
        arias = np.array([arias_NS, arias_EW,arias_VER])
        
        self.IAr = np.max([arias_VER, arias_NS, arias_EW])        
        
        arias_5 = 0.05 * self.IAr
        arias_95 = 0.95 * self.IAr
        
        delta_5 = np.abs(arias_5 - arias)
        result_5 = np.where(delta_5 == np.amin(delta_5))
        self.tiempo_5arias = result_5[1] * self.muestreo
        
        delta_95 = np.abs(arias_95 - arias)
        result_95 = np.where(delta_95 == np.amin(delta_95))
        self.tiempo_95arias = result_95[1] * self.muestreo
        
        self.duracionIA = self.tiempo_95arias - self.tiempo_5arias

        tiempoX = [self.tiempoV,self.tiempoV,self.tiempoV]
        ariasY = [arias_NS,arias_EW,arias_VER]

        self.plot(tiempoX,ariasY,'Tiempo, [s]','Intensidad de Arias [m/s]')
       
    def espectro_respuestaA(self,boolean_plot):
        
        # Parametros de entrada
        osc_damping = 0.05
        osc_freqs = np.logspace(-1, 2, 100)
        time_step = self.muestreo
        
        # Determinacion de los espectros de respuesta de aceleracion para cada componente
        
        self.osc_respsNS = calc_spec_accels(
                time_step, self.NS, osc_freqs, osc_damping
        )
        self.osc_respsEW = pyrotd.calc_spec_accels(
                time_step, self.EW, osc_freqs, osc_damping
        )      
        
        self.osc_respsVER = pyrotd.calc_spec_accels(
                time_step, self.VER, osc_freqs, osc_damping
        )

        # Se determinan los periodos para las graficas
        self.periodoNS = 1 / self.osc_respsNS.osc_freq
        self.periodoEW = 1 / self.osc_respsEW.osc_freq
        self.periodoVER = 1 / self.osc_respsVER.osc_freq

        self.osc_respsNS.spec_accel        
        self.osc_respsEW.spec_accel
        self.osc_respsVER.spec_accel
        
        # Se determinan espectros de velocidades

        self.PSV_NS = self.osc_respsNS.spec_accel / self.osc_respsNS.osc_freq
        self.PSV_EW = self.osc_respsEW.spec_accel / self.osc_respsEW.osc_freq
        self.PSV_VER = self.osc_respsVER.spec_accel / self.osc_respsVER.osc_freq
        
        
        #PSD = PSV / osc_respsNS.osc_freq
        self.PSD_NS = self.PSV_NS / self.osc_respsNS.osc_freq
        self.PSD_EW = self.PSV_EW / self.osc_respsEW.osc_freq
        self.PSD_VER = self.PSV_VER / self.osc_respsVER.osc_freq

        periodosX = [self.periodoNS,self.periodoEW,self.periodoVER]
        aceleracionesS = [self.osc_respsNS.spec_accel,self.osc_respsEW.spec_accel,self.osc_respsVER.spec_accel]
        velocidadS = [self.PSV_NS,self.PSV_EW,self.PSV_VER]
        desplazamientoS = [self.PSD_NS,self.PSD_EW,self.PSD_VER]

        if boolean_plot == True:

            self.plotGeneral(periodosX,aceleracionesS,'Periodo, [s]', 'Aceleracion espectral, $[m/s^2]$',3)
            self.plotGeneral(periodosX,velocidadS,'Periodo, [s]', 'Velocidad espectral, $[m/s]$',3)
            self.plotGeneral(periodosX,desplazamientoS,'Periodo, [s]', 'Desplazamiento espectral, $[m]$',5)
        else:
            pass

    def benioff(self):

        self.espectro_respuestaA(boolean_plot=False)
        
        self.benioff_pNS = integrate.cumtrapz(self.PSD_NS,self.osc_respsNS.osc_freq)
        self.benioff_pEW = integrate.cumtrapz(self.PSD_EW,self.osc_respsEW.osc_freq)
        self.benioff_pVER = integrate.cumtrapz(self.PSD_VER,self.osc_respsVER.osc_freq)

    def housner(self):

        self.espectro_respuestaA(boolean_plot=False)
        
        # Buscamos la posicion de T = 0.1 y T= 2.5
        
        delta01 = np.abs(0.1 - self.periodoNS)
        resultNS01 = np.where(delta01 == np.amin(delta01))
        limite_01 = np.max(resultNS01[0])
                
        delta25 = np.abs(2.5 - self.periodoNS)
        resultNS25 = np.where(delta25 == np.amin(delta25))
        limite_25 = np.max(resultNS25[0])

        # Arreglo recortado para la integracion
        
        periodoH = self.periodoNS[limite_25:limite_01]
        PSV_NS_H = self.PSV_NS[limite_25:limite_01]
        PSV_EW_H = self.PSV_EW[limite_25:limite_01]
        PSV_VER_H = self.PSV_VER[limite_25:limite_01]
        
        self.housner_PSV_NS = np.max(np.abs(integrate.cumtrapz(PSV_NS_H, periodoH)))
        self.housner_PSV_EW = np.max(np.abs(integrate.cumtrapz(PSV_EW_H, periodoH)))
        self.housner_PSV_VER = np.max(np.abs(integrate.cumtrapz(PSV_VER_H, periodoH)))

        print(self.housner_PSV_NS,self.housner_PSV_EW,self.housner_PSV_VER)

                
        
    def fourier(self):
        
        delta= self.muestreo
        npts = self.numero_datos
        self.spec, freqs = rfft(self.NS), rfftfreq(npts, 0.005)
        self.DFAS = self.spec * self.duracion
        self.Pw = self.DFAS**2 /(np.pi*self.duracion)
        
        print(len(freqs),len(self.DFAS))
        print(freqs,self.DFAS)
        
        fig= plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.loglog(freqs, np.abs(self.DFAS), label="raw FFT", color="red") #espectro crudo
        ax.set_ylim(0.001,np.max(self.DFAS).real*2)
        ax.set_title('FFT')
        ax.set_xlabel('Frequency, Hz')
        ax.set_ylabel('Fourier Amplitude, m/s/Hz')
        ax.grid()
        ax.legend()
        plt.show()
        
    def energy_density(self):
        
        C = 4500
        r = self.dist_hipocentral
        dens_material = 20
  
        y_integrate_NS = self.vel_NS**2
        y_integrate_EW = self.vel_EW**2
        y_integrate_VER = self.vel_VER**2
        
        integral_v2_NS = np.max(integrate.cumtrapz(y_integrate_NS,self.tiempoV))
        integral_v2_EW = np.max(integrate.cumtrapz(y_integrate_EW,self.tiempoV))
        integral_v2_VER = np.max(integrate.cumtrapz(y_integrate_VER,self.tiempoV))
     
        #Determinacion parametros energia NS
        self.Eo_NS = 4*np.pi*(r*1000)**2*dens_material*C*integral_v2_NS
        self.E_NS = self.Eo_NS / (4*np.pi*r**2)
        self.E_free_NS = np.max(self.E_NS / 4)
        self.flux_energy_NS = dens_material*C*y_integrate_NS / 4
        self.flux_energy_max_NS = np.max(self.flux_energy_NS)
        
        #Determinacion parametros energia EW
        self.Eo_EW = 4*np.pi*(r*1000)**2*dens_material*C*integral_v2_EW
        self.E_EW = self.Eo_EW / (4*np.pi*r**2)
        self.E_free_EW = np.max(self.E_EW / 4)
        self.flux_energy_EW = dens_material*C*y_integrate_EW / 4
        self.flux_energy_max_EW = np.max(self.flux_energy_EW)
        
        #Determinacion parametros energia Vertical
        self.Eo_VER = 4*np.pi*(r*1000)**2*dens_material*C*integral_v2_VER
        self.E_VER = self.Eo_VER / (4*np.pi*r**2)
        self.E_free_VER = np.max(self.E_VER / 4)
        self.flux_energy_VER = dens_material*C*y_integrate_VER / 4
        self.flux_energy_max_VER = np.max(self.flux_energy_VER)
        
        flux_energyY = [self.flux_energy_NS,self.flux_energy_EW,self.flux_energy_VER]
        tiempo_fluxX = [self.tiempoV,self.tiempoV,self.tiempoV]

        self.plot(tiempo_fluxX,flux_energyY,'Tiempo, [s]','Flujo de energia')

        
    def VDV(self):
        
        y_integrate_EW = self.EW**4      
        # Se integra con la ley del trapecio
        self.vdv = ((integrate.cumtrapz(y_integrate_EW,self.tiempo)))**(1/4)

        print(self.vdv)
        #print(self.tiempo.shape)

        self.plot(self.tiempo[0:len(self.vdv-1)],self.vdv,'Tiempo, [s]','VDV')
        
        plt.figure(figsize=(8,5))
        plt.plot(self.tiempo[0:len(self.vdv-1)],self.vdv, label="NS")
        plt.plot()
        plt.grid()
        plt.legend()
        plt.xlabel("tiempo (s)")
        plt.ylabel("Flujo de energia")
        plt.plot()
        plt.show()


if __name__ == "__main__":

    X = MovimientoFuerte("19990125181918_CFLAN.anc",'spline')
    print(X.dist_epicentral)
    X.housner()

    #X.energy_density()
    #X.flux_energy_EW
    #X.VDV()

