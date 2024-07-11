# this is an official Python runtime, used as the parent image
FROM node:14.17.4

WORKDIR /app

RUN node --version

COPY package.json ./

RUN npm install -g @angular/cli

RUN npm install

COPY . .

RUN ng build

# execute the Flask app
CMD ["node", "app.js"]
