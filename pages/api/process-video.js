import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import formidable from 'formidable';

// Import your machine learning model library and load your model
import { loadLayersModel } from '@tensorflow/tfjs-node';

// Define a global variable to hold the loaded model
let model;

// Load the model
async function loadModel() {
    model = await loadLayersModel('/model_vgg19.h5');
}

// Call the loadModel function when the server starts
loadModel();

export default async function handler(req, res) {
    if (req.method === 'POST') {
        try {
            const form = new formidable.IncomingForm();

            form.parse(req, async (err, fields, files) => {
                if (err) {
                    throw new Error('Failed to parse form data');
                }

                const videoFile = files.file;

                if (!videoFile) {
                    return res.status(400).json({ error: 'No file uploaded' });
                }

                // Execute the Python script as a child process
                const pythonProcess = spawn('python', ['/preprocessing.py', videoFile.path]);

                pythonProcess.stdout.on('data', (data) => {
                    console.log(`stdout: ${data}`);
                });

                pythonProcess.stderr.on('data', (data) => {
                    console.error(`stderr: ${data}`);
                });

                pythonProcess.on('close', (code) => {
                    console.log(`child process exited with code ${code}`);
                    res.status(200).json({ message: 'Script executed successfully' });
                });

            });
        } catch (error) {
            console.error('Error:', error);
            res.status(500).json({ error: 'Failed to execute script' });
        }
    } else {
        res.setHeader('Allow', ['POST']);
        res.status(405).end(`Method ${req.method} Not Allowed`);
    }
}
