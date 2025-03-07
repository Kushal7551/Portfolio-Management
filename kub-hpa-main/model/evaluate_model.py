from flask import Flask, request, jsonify
import os
import pandas as pd
import subprocess
    
app = Flask(__name__)

@app.route('/predict', methods=['POST','GET'])
def predict(): 
    data = request.json
    input_file = data.get('input_file')
    print(input_file)
    # if not input_file or not os.path.exists(input_file):
    #     return jsonify({'error': 'Invalid or missing input file'}), 400

    # Derive output file path
    output_file = input_file.replace('.csv', '_results.csv')

    try:
        # Run the inference script
        command = f"python inference.py app/{input_file}"
        result = subprocess.run(command, shell=True, capture_output=True)

        if result.returncode != 0:
            return jsonify({'error': result.stderr.decode()}), 500

        return jsonify({'output_file': output_file})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)