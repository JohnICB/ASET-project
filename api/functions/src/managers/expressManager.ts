import express from 'express';
import * as cors from 'cors';
import { LoggingHandler } from '../handlers/loggingHandler';
import { inputRouter } from '../routes/inputRouter';
import { AuthorizationHandler } from '../handlers/authorizationHandler';

export class ExpressManager
{
    public ConfigureExpress(app: express.Express): void
    {
        app.use(express.json());
        app.use(express.urlencoded({ extended: false }));

        app.disable('x-powered-by');

        app.use(cors.default());

        app.use(LoggingHandler.LogRequest);
        app.use(AuthorizationHandler.HandleAuthorizationCheck);

        app.use(inputRouter);
    }
}

