# Dockerfile for React frontend

# Build react frontend
FROM node:18

# Working directory be app
WORKDIR /app


COPY package*.json ./

###  Installing dependencies

RUN npm install

# copy local files to app folder
COPY . .

EXPOSE 5173

CMD ["npm","run","dev"]