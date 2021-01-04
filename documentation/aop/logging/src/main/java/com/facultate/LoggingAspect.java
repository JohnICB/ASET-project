package com.facultate;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.After;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.stereotype.Component;
import org.springframework.util.StopWatch;

@Aspect
@Component
public class LoggingAspect 
{
	private static final Logger LOGGER = LogManager.getLogger(LoggingAspect.class);

    private String methodSignature;
    private StopWatch stopWatch;

    @Before("execution(* com.facultate.VideoPreprocessor..*(..)))")
    public void startProfiling(JoinPoint thisJoinPoint) throws Throwable
    {
        SaveMethodSignature(thisJoinPoint);

        StartTimer();
    }

    private void StartTimer()
    {
        this.stopWatch = new StopWatch();
        stopWatch.start();
    }

    @After("execution(* com.facultate.VideoPreprocessor..*(..)))")
    public void endProfiling(JoinPoint thisJoinPoint) throws Throwable
    {
        stopWatch.stop();
        LOGGER.info(methodSignature + " :: " + stopWatch.getTotalTimeMillis() + " ms");
    }

    private void SaveMethodSignature(JoinPoint thisJoinPoint)
    {
        MethodSignature methodSignature = (MethodSignature) thisJoinPoint.getSignature();

        String className = methodSignature.getDeclaringType().getSimpleName();
        String methodName = methodSignature.getName();

        this.methodSignature = "Execution time of " + className + "." + methodName;
    }
}
