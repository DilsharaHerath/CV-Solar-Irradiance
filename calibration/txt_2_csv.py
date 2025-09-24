# This file will transform txt file for timeseries to csv form 
import csv

def convert_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write header row
        writer.writerow(['Date-Time', 'GHI', 'DNI', 'DHI', 'Temp', 'Pressure'])
        
        in_data_section = False
        
        for line in infile:
            line = line.strip()
            if line == '#begindata':
                in_data_section = True
                continue
            if in_data_section and line:  # Process only non-empty lines after #begindata
                # Split by tabs and write to CSV
                data = line.split('\t')
                if len(data) >= 6:  # Ensure at least 6 columns
                    writer.writerow(data[:6])  # Take first 6 columns

# Usage
input_filename = './../dataset/PSA_timeSeries/PSA_timeSeries_Metas.txt'
output_filename = 'PSA_timeSeries_data2.csv'

convert_to_csv(input_filename, output_filename)
print(f"Conversion complete! Data CSV file saved as {output_filename}")