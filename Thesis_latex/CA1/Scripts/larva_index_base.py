import numpy as np

def iterate(t, dt, n, lfood, frc, wtol, eff, mc, total_waste, larvae, fQ, var):
    np.random.shuffle(larvae)
    vf_pre, vf_post = 1, 1.5
    u = 10**4
    food_aq = var['food_aq']
    waste_prod = var['waste_prod']

    for i in larvae:
        if lfood > total_waste and fQ > 0 :
            if var['size'][i] < mc[i]:
                if var['size'][i] != -1 :
                    var['frt'][i] = frc[i] + 0.017 * t
                    vfood = vf_pre * dt
                    incr = var['frt'][i] * vfood * fQ * eff[i]
                    if var['w_acum'][i] > u * wtol[i]:
                        var['size'][i] = -1
                    else:
                        var['size'][i] += incr
                        var['w_acum'][i] +=  var['frt'][i] * vfood * (1 - fQ)
                        waste_prod[i] = var['frt'][i] * vfood * fQ * (1 - eff[i])
                        food_aq[i] = var['frt'][i] * vfood
                        var['dt'][i] += dt
            else:
                if t < var['dt'][i] + 48*60:
                    vfood = vf_post * dt
                    incr = var['frt'][i] * vfood * fQ * eff[i]
                    if var['w_acum'][i] <= u * wtol[i]:
                        var['size'][i] += incr
                        var['w_acum'][i] +=  var['frt'][i] * vfood * (1 - fQ)
                        waste_prod[i] = var['frt'][i] * vfood * fQ * (1 - eff[i])
                        food_aq[i] = var['frt'][i] * vfood
                else:
                    break
        elif var['size'][i] < mc[i]:
            var['size'][i] = -1

    return var, larvae

def foodQ(kd, lfood, food_aq, waste_prod, fQ, waste):
    fband = 50*4*(35.5*10**5)/8
    dband = fband
    if lfood > fband+dband:
        waste['fband'] = (waste['fband']+waste_prod)*(1-kd) + food_aq*(1-fQ['dband'])
        waste['dband'] += kd*(waste['fband']+waste_prod) - food_aq*(1-fQ['dband'])
    elif lfood > fband:
        waste['fband'] = (waste['fband']+waste_prod) + food_aq*(1-fQ['dband'])
        waste['dband'] -= food_aq*(1-fQ['dband'])
    else:
        fband = lfood
        waste['dband'] = 0
        waste['fband'] += waste_prod
    fQ['fband'] = 1 - waste['fband']/fband
    fQ['dband'] = 1 - waste['dband']/dband

    return fQ, waste

def survive(index, mc, wtol, var):
    u = 0.000007
    ldead = np.zeros(2)
    alive_size = []
    alive_devt = []
    alive_index = []
    alive_fr = []
    for i in index:
        if var['size'][i] < mc[i]:
            ldead[0] += 1
        elif np.random.uniform() < 1-np.exp(-(u*var['w_acum'][i])**2):
            ldead[1] += 1
        else:
            alive_size += [var['size'][i]]
            alive_devt += [var['dt'][i]/60]
            alive_index += [i]
            alive_fr += [var['frt'][i]]
    return ldead, alive_size, alive_devt, alive_index, alive_fr

def larval_growth(lfood, m_index, f_index, m_traits, f_traits):
    nf = len(f_index)
    nm = len(m_index)
    m_var = dict(size = np.repeat(3., nm), dt = np.zeros(nm), frt = np.zeros(nm), w_acum = np.zeros(nm), food_aq = np.zeros(nm), waste_prod = np.zeros(nm))
    f_var = dict(size = np.repeat(3., nf), dt = np.zeros(nf), frt = np.zeros(nf), w_acum = np.zeros(nf), food_aq = np.zeros(nf), waste_prod = np.zeros(nf))
    feff, meff = f_traits['eff']*0.011, m_traits['eff']*0.009
    fmc, mmc = f_traits['mc']*1.1, m_traits['mc']*.9
    dt = 30
    fQ = dict(fband = 1., dband = 1.)
    waste = dict(fband = 0, dband = 0)
    total_waste = 0
    kd = 0.002*dt
    lt, lgraph = [], []
    for t in np.arange(0,105*60,dt):
        if lfood <= total_waste:
            break
        fout = iterate(t, dt, nf, lfood, f_traits['fr'], f_traits['wtol'], feff, fmc, total_waste, f_index, fQ['fband'], f_var)
        mout = iterate(t, dt, nm, lfood, m_traits['fr'], m_traits['wtol'], meff, mmc, total_waste, m_index,fQ['fband'], m_var)
        food_aq = sum(f_var['food_aq']) + sum(m_var['food_aq'])
        waste_prod = sum(f_var['waste_prod']) + sum(m_var['waste_prod'])
        lfood -= (food_aq - waste_prod)
        foodQ(kd, lfood, food_aq, waste_prod, fQ, waste)
        total_waste += waste_prod

        lt += [t/60]
        lgraph += [fQ['fband']]

    falive_out = survive(fout[1], fmc, f_traits['wtol'], f_var)
    malive_out = survive(mout[1], mmc, m_traits['wtol'], m_var)

    survivors = (nf-sum(falive_out[0]) + nm-sum(malive_out[0]))*.98/(nf+nm)
    alive_size = dict(female = falive_out[1], male = malive_out[1])
    alive_devt = dict(female = falive_out[2], male = malive_out[2])
    alive_index = dict(female = falive_out[3], male = malive_out[3])
    alive_frt = dict(female = falive_out[4], male = malive_out[4])
    
    return alive_size, alive_devt, alive_frt, survivors, alive_index, lt, lgraph
