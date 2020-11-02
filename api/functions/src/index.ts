import express from 'express';
import { ExpressManager } from './managers/expressManager';
import * as functions from 'firebase-functions';

const app: express.Express = express();
new ExpressManager().ConfigureExpress(app);

const deployRegion: string = 'europe-west6';
export const api = functions.region(deployRegion).https.onRequest(app);
