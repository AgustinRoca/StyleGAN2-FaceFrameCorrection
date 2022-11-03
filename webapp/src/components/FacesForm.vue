<template>
<v-container style="width: 70vw">
    <v-form v-model="valid" @submit="submit">
        <v-text-field
            v-model="amount"
            :rules="amountRules"
            label="Amount"
            required
            type="number"
        ></v-text-field>
        <v-btn
        color="primary"
        class="mr-4"
        type="submit"
        :disabled="isLoading"
        >
        Generate
        </v-btn>
        <v-progress-circular
                    indeterminate
                    color="primary"
                    v-if="isLoading"
        ></v-progress-circular>
    </v-form>
</v-container>
</template>

<script>
    import axios from 'axios'
    export default {
        data: () => ({
        refCount: 0,
        isLoading: false,
        valid: false,
        amount: '',
        amountRules: [
            v => !!v || 'Amount is required',
            v => Number.isInteger(v) || 'Amount must be an integer',
            v => Number(v) > 0 || 'Amount must be a positive integer',
        ],
        }),
        methods : {
            submit(){
                let formData = new FormData();
                formData.append('amount', this.amount)
                axios({
                    url: 'https://pf-2021-generadorcaras.it.itba.edu.ar/api/faces',
                    method: 'post',
                    data: formData,
                    headers: { "Content-Type": "multipart/form-data" },
                }).then((response) => {
                    this.$emit('generated', response.data.imgs_bytes, response.data.zs)
                }).catch((e) => {
                    alert('Error')
                    console.log(e)
                })
            },
            setLoading(isLoading) {
                if (isLoading) {
                    this.refCount++;
                    this.isLoading = true;
                } else if (this.refCount > 0) {
                    this.refCount--;
                    this.isLoading = (this.refCount > 0);
                }
            }
        },
        created() {
            axios.interceptors.request.use((config) => {
                this.setLoading(true);
                return config;
            }, (error) => {
                this.setLoading(false);
                return Promise.reject(error);
            });

            axios.interceptors.response.use((response) => {
                this.setLoading(false);
                return response;
            }, (error) => {
                this.setLoading(false);
                return Promise.reject(error);
            });
        }
    }
</script>