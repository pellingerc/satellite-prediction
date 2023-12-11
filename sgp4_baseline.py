from sgp4.api import Satrec
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
import numpy as np

def create_sgp4_predictions(file_path, target_steps):
    '''
    creates baseline predictions using sgp4
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()
        #last 15%
        lines = lines[(int(0.85*len(lines))):]

    prediction = np.zeros(((len(lines) // 2) - (target_steps), target_steps, 6))
    base = np.zeros(((len(lines) // 2) - (target_steps), target_steps, 6))
    for i in range(0, len(lines) - (target_steps*2), 2):
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()
        assert twoline2rv(lines[i], lines[i + 1], wgs72)
        satellite = Satrec.twoline2rv(lines[i], lines[i + 1])

        for j in range(target_steps):
            futute_pos = Satrec.twoline2rv(lines[i + (2*j)], lines[i + 1 + (2*j)])
            bit, position, velocity = satellite.sgp4(futute_pos.jdsatepoch, futute_pos.jdsatepochF + j)
            true_bit, true_position, true_velocity = futute_pos.sgp4(futute_pos.jdsatepoch, futute_pos.jdsatepochF + j)

            if position[0] == np.nan or position[1] == np.nan or position[2] == np.nan:
                print('nan found at index: ', i)
            
            if true_position[0] == np.nan or true_position[1] == np.nan or true_position[2] == np.nan:
                print('nan found at index: ', i)

            if velocity[0] == np.nan or velocity[1] == np.nan or velocity[2] == np.nan:
                print('nan found at index: ', i)
            
            if true_velocity[0] == np.nan or true_velocity[1] == np.nan or true_velocity[2] == np.nan:
                print('nan found at index: ', i)

           
            prediction[i // 2, j, 0] = position[0]
            prediction[i // 2, j, 1] = position[1]
            prediction[i // 2, j, 2] = position[2]
            prediction[i // 2, j, 3] = velocity[0]
            prediction[i // 2, j, 4] = velocity[1]
            prediction[i // 2, j, 5] = velocity[2]
            base[i // 2, j, 0] = true_position[0]
            base[i // 2, j, 1] = true_position[1]
            base[i // 2, j, 2] = true_position[2]
            base[i // 2, j, 3] = true_velocity[0]
            base[i // 2, j, 4] = true_velocity[1]
            base[i // 2, j, 5] = true_velocity[2]
            

    return prediction, base

        
