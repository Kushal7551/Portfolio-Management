const express = require('express');
const fs = require('fs');
const csvParser = require('csv-parser');
const cors = require('cors');
const bodyParser = require('body-parser'); 
const axios = require('axios');
const { log } = require('console');
const { exec } = require('child_process');
const path = require('path');


const app = express();
const PORT = 3000;
 
const corsOptions = { 
    origin:"http://192.168.58.2:30008",
    credentials:true
}
app.use(cors(corsOptions));
app.use(bodyParser.json()); 


// const MODEL_HOST = process.env.MODEL_HOST || 'myapp-model';
// const MODEL_HOST = process.env.MODEL_HOST || 'myapp-model';
const MODEL_PORT = process.env.MODEL_PORT || 4000;

app.post('/read-csv', async (req, res) => {
    const { fileName } = req.body;
    console.log(fileName);
    
    if (!fileName) {
        return res.status(400).json({ error: 'fileName is required' });
    }

    const inputFilePath = path.join('dataset', fileName);
    let outputFilePath = path.join('dataset', fileName.replace('.csv', '_results.csv'));


    console.log(inputFilePath);
    // const response = await axios.post(`http://192.168.58.2:${MODEL_PORT}/predict`, {
        const response = await axios.post(`http://192.168.58.2:30009/predict`, {
        input_file: inputFilePath
    });
//     const response = await axios.post('http://192.168.58.2:30009/predict', {
//     input_file: inputFilePath
// }, {
//     headers: {
//         'Content-Type': 'application/json'
//     }
// });
    console.log(response);
    console.log(outputFilePath);
    
    const results = [];
    outputFilePath = `/app/${outputFilePath}`
    fs.createReadStream(outputFilePath)
        .pipe(csvParser())
        .on('data', (data) => results.push(data))
        .on('end', () => {
            res.json(results); 
        })
        .on('error', (err) => {
            res.status(500).json({ error: 'Error reading the processed CSV file', details: err.message });
        });
});     
const { spawn } = require('child_process');
const runPythonScript = (inputData) => {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, 'sentimentAnalysis.py');
        // const scriptPath = path.join(__dirname, 'model-1', 'sentimentAnalysis.py');
        // console.log(scriptPath);
        const pythonProcess = spawn('python3',[scriptPath]); 
        pythonProcess.stdin.write(JSON.stringify(inputData));
        pythonProcess.stdin.end();
        let result = '';
        let errorOutput = '';
        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Python script exited with code ${code}: ${errorOutput}`));
            } else {
                try {
                    resolve(JSON.parse(result)); // Parse the JSON result from Python
                } catch (error) {
                    reject(new Error(`Failed to parse Python script output: ${result}`));
                }
            }
        });
    });
};


const input = { number: 5 };

app.post('/sample', async (req, res) => {
    const fileName = 'stock_sentimentScore.csv';
    // const filePath = path.join(__dirname, 'model-1', fileName);
    const filePath = path.join(__dirname, fileName);

    const checkFileInterval = 3000; // Set the interval to 3 seconds
    const maxAttempts = 10; // Maximum number of polling attempts
    let attempts = 0;

    // Function to periodically check for the file
    const pollForFile = async () => {
        return new Promise((resolve, reject) => {
            const intervalId = setInterval(() => {
                attempts++;
                if (fs.existsSync(filePath)) {
                    console.log(`File found after ${attempts} attempt(s): ${fileName}`);
                    clearInterval(intervalId);
                    resolve(true);
                } else if (attempts >= maxAttempts) {
                    console.error(`File not found after ${maxAttempts} attempt(s).`);
                    clearInterval(intervalId);
                    reject(new Error(`File ${fileName} not created after ${maxAttempts} attempts.`));
                } else {
                    console.log(`File not found: ${fileName}. Retrying in ${checkFileInterval / 1000} seconds...`);
                }
            }, checkFileInterval);
        });
    };

    // Main logic for the endpoint
    try {
        if (!fs.existsSync(filePath)) {
            console.log(`File not found: ${fileName}. Running Python script...`);
            const pythonOutput = await runPythonScript(input);
            console.log('Python script output:', pythonOutput);

            // Save Python output as a CSV file
            fs.writeFileSync(filePath, JSON.stringify(pythonOutput, null, 2));

            // Poll for file existence
            await pollForFile();
        }

        // File exists, parse and return its content
        const results = [];
        fs.createReadStream(filePath)
            .pipe(csvParser())
            .on('data', (data) => results.push(data))
            .on('end', () => {
                res.json(results);
            })
            .on('error', (err) => {
                res.status(500).json({ error: 'Error reading the CSV file', details: err.message });
            });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// if (require.main === module) {
//     // Only start the server if this file is executed directly
//     app.listen(PORT, () => {
//         console.log(`Server is running on http://localhost:${PORT}`);
//     });
// }
if (require.main === module) {
    // Only start the server if this file is executed directly
    app.listen(PORT, () => {
        console.log(`Server is running on http://192.168.58.2:${PORT}`);
    });
}

module.exports = app; // Export the app for testing
