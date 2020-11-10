
import express, { Router } from 'express';
import { InputController } from '../controllers/inputController';

export const inputRouter: Router = express.Router();

inputRouter.post('/', InputController);
