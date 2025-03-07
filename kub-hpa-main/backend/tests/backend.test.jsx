const request = require('supertest');
const fs = require('fs');
const path = require('path');
const app = require('./app'); // Import the app from the main file

jest.mock('axios');
const axios = require('axios');

describe('Backend API Tests', () => {
    // Mock file system and axios behavior where needed
    afterAll(() => {
        jest.restoreAllMocks(); // Clean up any mocks
    });

    // test('POST /read-csv should return processed data from the CSV file', async () => {
    //     // Mock axios response
    //     axios.post.mockResolvedValue({ data: 'Processing Complete' });

    //     // Mock existence of result CSV file
    //     const mockFilePath = path.join('dataset', 'test_results.csv');
    //     fs.existsSync = jest.fn().mockReturnValue(true);
    //     fs.createReadStream = jest.fn().mockImplementation(() => {
    //         const { Readable } = require('stream');
    //         const mockStream = new Readable();
    //         mockStream.push('column1,column2\nvalue1,value2\n');
    //         mockStream.push(null); // End of stream
    //         return mockStream;
    //     });

    //     const response = await request(app)
    //         .post('/read-csv')
    //         .send({ fileName: 'test.csv' });

    //     expect(response.statusCode).toBe(200);
    //     expect(response.body).toEqual([{ column1: 'value1', column2: 'value2' }]);
    //     expect(fs.createReadStream).toHaveBeenCalledWith(`/app/${mockFilePath}`);
    // });

    test('POST /read-csv should return 400 if fileName is missing', async () => {
        const response = await request(app).post('/read-csv').send({});
        expect(response.statusCode).toBe(400);
        expect(response.body).toEqual({ error: 'fileName is required' });
    });

    test('POST /sample should process the input and return CSV content', async () => {
        // Mock file system behavior for checking file existence
        const mockFilePath = path.join(__dirname, 'stock_sentimentScore.csv');
        fs.existsSync = jest.fn((filePath) => filePath === mockFilePath);

        // Mock createReadStream behavior
        fs.createReadStream = jest.fn(() => {
            const { Readable } = require('stream');
            const mockStream = new Readable();
            mockStream.push('col1,col2\nval1,val2\n');
            mockStream.push(null); // End of stream
            return mockStream;
        });

        const response = await request(app).post('/sample').send();

        expect(response.statusCode).toBe(200);
        expect(response.body).toEqual([{ col1: 'val1', col2: 'val2' }]);
        expect(fs.createReadStream).toHaveBeenCalledWith(mockFilePath);
    });
});
