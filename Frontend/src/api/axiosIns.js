import axios from 'axios';

const axiosInstance = axios.create({
  baseURL:'https://agriyield-predictor-04gw.onrender.com',
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default axiosInstance;
