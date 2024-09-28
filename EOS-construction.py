import numpy as np

def energy_density(n,k,gamma):

    baryon_mass = 931.2 # MeV   
    n0 = 0.135

    energy = baryon_mass*n + (k*baryon_mass*n0)/(gamma-1)*((n/n0)**gamma)

    return energy

def calc_pressure(n,k,gamma):

    n0 = 0.135
    baryon_mass = 931.2 # MeV   

    pressure = k*baryon_mass*n0*((n/n0)**gamma)

    return pressure

def sound_speed(p2,p1,e2,e1):

    c2 = (p2 - p1)/(e2 - e1)

    return c2


if __name__ == '__main__':

    k = 0.038
    gamma = 2.0

    # n = np.arange(1.0e-7, 0.12, (0.120-1.0e-7)/1021)

    n1 = np.arange(0.01,1.5 ,0.05)

    energy = energy_density(n1,k,gamma)
    pressure = calc_pressure(n1,k,gamma)

    # print('Energy density:', energy)
    # print('Pressure:', pressure)

    c2 = sound_speed(pressure[1:],pressure[:-1],energy[1:],energy[:-1])
    # print('Sound speed:', c2)

    # print((pressure[2] - pressure[1])/(energy[2] - energy[1]))

    print(n1)

    with open(f'Polytropic-Eos-gamma_new-new{gamma}.dat', 'w') as f:
        #f.write('NumberDensity','Energy_Density','Pressure','Sound_Speed\n')

        for i in range(len(n1)-1):
                            # energy = energy_density(n,k,gamma)
                            # pressure = calc_pressure(n,k,gamma)
                            # c2 = sound_speed(pressure[1:],pressure[:-1],energy[1:],energy[:-1])
                            f.write(f'{n1[i]},{energy[i]},{pressure[i]},{c2[i]}  \n ' )
        
    # with open(f'Polytropic-Eos-gamma_new{gamma}.dat', 'a') as f:
    #     for i in range(len(n1)):

    #                         energy = energy_density(n1,k,gamma)
    #                         pressure = calc_pressure(n1,k,gamma)
    #                         c2 = sound_speed(pressure[1:],pressure[:-1],energy[1:],energy[:-1])
    #                         f.write(f'{pressure[i]} \t {energy[i]} \t {c2[i]} \t {n1[i]}  \n ' )

                    
    
    print(f'EOS data written to file: Polytropic-Eos-gamma_new_new{gamma}.dat')

