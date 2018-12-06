#Pkg.add("Clp")
#Pkg.add("JuMP")
#Pkg.add("Plots")
#Pkg.add("PlotlyJS")
#Pkg.add("GLPKMathProgInterface")
#Pkg.add("DataFrames")
#Pkg.add("Distributions")
#Pkg.add("StatPlots")
#Pkg.update()
#Pkg.add("CSV")

using JuMP, Plots, GLPKMathProgInterface, DataFrames,
    Distributions, Random, StatPlots, CSV
Path = dirname(Base.source_path());
cr = 4000 #deficit cost

#Reading data from files--------------------------------------------------------
#Generation data
IN_DATA = CSV.read(Path * "/INPUT_GEN_IEEE_2.csv");
nGen    = size(IN_DATA)[1]
G       = convert(Array{Float64,1},IN_DATA[1:nGen,1]) #Gen. capacity (MW)
c       = convert(Array{Float64,1},IN_DATA[1:nGen,2]) #Gen. costs ($/MWh)
Gbus    = convert(Array{Int64,1},IN_DATA[1:nGen,3]) #Bus address
usina = 8 #Gerador do qual se deseja maximizar a receita
# -----------------------------------------------------------------------------------

#Transmission data
IN_DATA = CSV.read(Path * "/INPUT_LIN_IEEE_2.csv");
nLin    = size(IN_DATA)[1]
F       = convert(Array{Float64,1},IN_DATA[1:nLin,1]) #Line capacity (MW)
x       = convert(Array{Float64,1},IN_DATA[1:nLin,2]) #Line reactance (Ω)
from    = convert(Array{Int16,1},IN_DATA[1:nLin,3]) #Line from (a bus)
to      = convert(Array{Int16,1},IN_DATA[1:nLin,4]) #Line to (a bus)

#Buses data
IN_DATA = CSV.read(Path * "/INPUT_BUS_IEEE_2.csv");
nBus    = size(IN_DATA)[1]
d_perc  = convert(Array{Float64,1},IN_DATA[1:nBus,1]) #Bus demand (%)

#Demand data
IN_DATA = CSV.read(Path * "/INPUT_DEMAND_v2.csv");
T       = size(IN_DATA)[1]
dh      = convert(Array{Float64,1},IN_DATA[1:T,1]) #Demand per month
d       = zeros(nBus,T)

#Calculando Demanda por barra
for i in 1:nBus
    d[i,:] = d_perc[i] .* dh
end

function main(d)

    #Optimization model - cooptimization of energy ---------------------
    dispatchmodel = Model(solver=GLPKSolverLP()); #Defines the optimization model
    #Defining decision variables
    @variable(dispatchmodel, g[i=1:nGen,t=1:T]>=0);
    @variable(dispatchmodel, f[l=1:nLin,t=1:T]);
    @variable(dispatchmodel, θ[b=1:nBus,t=1:T]);
    @variable(dispatchmodel, deficit[i=1:nBus, t=1:T]>=0); #deficit added

      #Defining constraints
      # --------------------------
    @constraints(dispatchmodel, begin
        Kirchhoff1pre[b=1:nBus, t=1:T], sum(g[i,t] for i in findall((collect(1:nGen).*(Gbus.==b)).>0)) +
                        sum(f[l,t] for l in findall((collect(1:nLin).*(to.==b)).>0)) -
                        sum(f[l,t] for l in findall((collect(1:nLin).*(from.==b)).>0)) == d[b,t] - deficit[b,t]
        GenMaxCapacity[i=1:nGen, t=1:T], g[i,t] <= G[i]
        GenMinCapacity[i=1:nGen, t=1:T], g[i,t] >= 0
        Kirchhoff2pre[i=1:nLin, t=1:T], f[i,t] == (θ[from[i],t] - θ[to[i],t]) / x[i]
        MaxTransCapacity[i=1:nLin, t=1:T], -F[i] <= f[i,t] <= F[i]
    end)

    #Objective function
    @objective(dispatchmodel,Min, sum(sum((c[i]*g[i,t]) for i=1:nGen) +
                sum(cr*deficit[j,t] for j=1:nBus) for t=1:T))

        #Solve model
    status = solve(dispatchmodel)
    #Getting decision variables
    if status == :Optimal
        gen      = getvalue(g)
        flow     = getvalue(f)
        ang      = getvalue(θ)
        Deficit  = getvalue(deficit)
        obj      = getobjectivevalue(dispatchmodel)
        cmo     =  getdual(Kirchhoff1pre)
    end
    return gen, cmo

end # function main

#--------------------------------------------------------
#Monte Carlo Simulation
nscen = 100 #number of scenarios
demand_scen = zeros(nscen,12)

for m=1:12
    Random.seed!(100+m)
    demand_scen[:,m] = rand(Normal(dh[m],315), nscen)
end

dtrans = transpose(demand_scen)
plot(dtrans, legend = false, ylim = (0,4000), xlim = (1,12), xticks = 1:1:12)

gen_scen = zeros(nscen,12)
cmo_scen = zeros(nscen,12)
cmo_tot = zeros(nscen*24,12)
#simulação de todos cenários de Demanda
@time for s = 1:nscen
    for i in 1:nBus
        d[i,:] = d_perc[i] .* dtrans[:,s]
    end
    global gen, cmo = main(d)
    gen_scen[s,:] = gen[usina,:]
    cmo_scen[s,:] = cmo[13,:]
    cmo_tot[((s-1)*24+1):s*24,:] = cmo[:,:]
end

plot(transpose(cmo_tot), legend = false,xlim = (1,12), xticks = 1:1:12)
plot(transpose(cmo_scen), legend = false,xlim = (1,12), xticks = 1:1:12)
CSV.write(Path * "\\Results\\cmo_tot.out", DataFrame(cmo_tot))
CSV.write(Path * "\\Results\\cmo_barra13.out", DataFrame(cmo_scen))

gen_scenall = vec(gen_scen)
cmo_scenall = vec(cmo_scen)
density(gen_scenall,xlim = (0,G[usina]+100), legend = false)
density(cmo_scenall, legend = false)
CSV.write(Path * "\\Results\\cmo_scen.out", DataFrame([1:1200,cmo_scenall]))
CSV.write(Path * "\\Results\\gen_scen.out", DataFrame([1:1200,gen_scenall]))

plot(transpose(gen_scen), legend = false, ylim = (0,G[usina]), xlim = (1,12), xticks = 1:1:12)
plot(transpose(cmo_scen), legend = false, ylim = (0,1000), xlim = (1,12), xticks = 1:1:12)

#-----------------------------------------------
# Construindo modelo de maximização de receita
# Primeiramente sem CVaR

P = mean(cmo_scenall) #Média do preço do spot
cvu = c[usina]

revenue_model = Model(solver=GLPKSolverLP())
@variable(revenue_model, 0 <= Q <= G[usina])
@variable(revenue_model, Z[i=1:nscen*12])
@constraint(revenue_model, Receita[s=1:nscen*12], Z[s] == (P-cmo_scenall[s])*Q +
    gen_scenall[s]*(cmo_scenall[s]-cvu))

@objective(revenue_model, Max, sum(Z[s] for s=1:nscen*12)/(nscen*12))

status_rev = solve(revenue_model)

Q_otimo = getvalue(Q)
Z_otimo = getobjectivevalue(revenue_model)
R_scen = getvalue(Z)
plot(R_scen)
CSV.write(Path * "\\Results\\R_alpha100.out", DataFrame([1:1200,R_scen]))

#-----------------------------------------------
# Construindo modelo de maximização de receita
# Co-otimizando Com CVaR

function portfolio_opt(α,λ,P)
    cvu = c[usina]
    revenue_opt = Model(solver=GLPKSolverLP())
    @variable(revenue_opt, δ[i=1:nscen*12] >= 0)
    @variable(revenue_opt, Z)
    @variable(revenue_opt, 0 <= Q <= G[usina])
    @variable(revenue_opt, R[i=1:nscen*12])
    @constraint(revenue_opt, Receita[s=1:nscen*12], R[s] == (P-cmo_scenall[s])*Q +
        gen_scenall[s]*(cmo_scenall[s]-cvu))
    @constraint(revenue_opt, Receita_CVAR[s=1:nscen*12], δ[s] >= Z - R[s])

    @objective(revenue_opt, Max, (1-λ)*sum(R[s] for s=1:nscen*12)/(nscen*12) +
                                λ*(Z - (sum(δ[s]/α for s=1:nscen*12)/(nscen*12))))

    status_rev_opt = solve(revenue_opt)

    Q_opt = getvalue(Q)
    Z_obj_opt = getobjectivevalue(revenue_opt)
    Receita = getvalue(R)
    mean_Receita = mean(Receita)
    return Q_opt, Z_obj_opt, Receita, mean_Receita
end

Q_50, Z_50, Receita_50, R50_mean = portfolio_opt(0.5,0.4,P)
Q_90, Z_90, Receita_90, R90_mean = portfolio_opt(0.9,0.4,P)
Q_100, Z_100, Receita_100, R100_mean = portfolio_opt(1,0.0,P)

CSV.write(Path * "\\Results\\R_alpha50.out", DataFrame([1:1200,Receita_50]))
CSV.write(Path * "\\Results\\R_alpha90.out", DataFrame([1:1200,Receita_90]))
CSV.write(Path * "\\Results\\R_alpha100.out", DataFrame([1:1200,Receita_100]))

#-----------------------------------------------
# Construindo modelo de maximização de receita
# Otimizando para diferentes parâmetros de alpha e P
tabela_neutro = zeros(41,2)
tabela_risco1 = zeros(41,2)
tabela_risco2 = zeros(41,2)

λ = 0.4
@time for α = 1:3
    if α == 2
        α = 0.9
        for P = 1:41
            P_ = P*5+145
            Q_90, Z_90, Receita_90 = portfolio_opt(α,λ,P_)
            tabela_risco2[P,1] = P_
            tabela_risco2[P,2] = Q_90/G[usina]
        end
    elseif α == 3
        α = 0.5
        for P = 1:41
            P_ = P*5+145
            Q_50, Z_50, Receita_50 = portfolio_opt(α,λ,P_)
            tabela_risco1[P,1] = P_
            tabela_risco1[P,2] = Q_50/G[usina]
        end
    else
        for P = 1:41
            P_ = P*5+145
            Q_neutro, Z_neutro, Receita_neutro = portfolio_opt(α,λ,P_)
            tabela_neutro[P,1] = P_
            tabela_neutro[P,2] = Q_neutro/G[usina]
        end
    end
end
CSV.write(Path * "\\Results\\tabela_90%.out", DataFrame(tabela_risco2))
CSV.write(Path * "\\Results\\tabela_50%.out", DataFrame(tabela_risco1))
CSV.write(Path * "\\Results\\tabela_neutro.out", DataFrame(tabela_neutro))
