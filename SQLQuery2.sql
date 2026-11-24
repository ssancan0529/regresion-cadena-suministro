USE PROJECT_OPT_NUM;
GO

-- Si ya existe, la eliminamos para recrearla
IF OBJECT_ID('dbo.supply_chain_regression', 'U') IS NOT NULL
    DROP TABLE dbo.supply_chain_regression;
GO

-- Crear tabla limpia para regresión
SELECT
    Price,
    Manufacturing_costs     AS manufacturing_costs,
    Shipping_costs          AS shipping_costs,
    Production_volumes      AS production_volumes,
    Manufacturing_lead_time AS manufacturing_lead_time
INTO dbo.supply_chain_regression
FROM dbo.supply_chain_model
WHERE
    Price IS NOT NULL
    AND Manufacturing_costs IS NOT NULL
    AND Shipping_costs IS NOT NULL
    AND Production_volumes IS NOT NULL
    AND Manufacturing_lead_time IS NOT NULL;
GO

SELECT TOP 10 * FROM supply_chain_model;

CREATE TABLE resultados_modelo (
    metodo VARCHAR(50),
    beta0 FLOAT,
    beta1 FLOAT,
    beta2 FLOAT,
    beta3 FLOAT,
    beta4 FLOAT,
    f_beta FLOAT,
    iteraciones INT
);
SELECT * FROM resultados_modelo;
SELECT COUNT(*) AS n
FROM dbo.supply_chain_regression;
